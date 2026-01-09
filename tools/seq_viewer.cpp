#include <string>
#include <vector>
#include <memory>
#include "igl/opengl/glfw/Viewer.h"
#include "portaudio.h"
#include "modal_solver.h"
#include "forces.h"
#include "config.h"
#include "ModalMaterial.h"
#include "igl/read_triangle_mesh.h"
#include "igl/per_vertex_normals.h"
#include "igl/opengl/glfw/imgui/ImGuiMenu.h"

/*------------------------------------
Creates a libigl viewer for simulation sequence with sound synthesis
*--------------------------------*/
#define CHECK_PA_LAUNCH(x) \
    { \
        auto pa_err = x; \
        if (pa_err != paNoError) \
            printf("**ERROR** PortAudio error: %s\n", Pa_GetErrorText(pa_err));\
    }
using namespace std;
using scalar = double;
using mat3 = Eigen::Matrix<scalar, 3, 3>;
using vec3 = Eigen::Matrix<scalar, 3, 1>;
using q4 = Eigen::Matrix<scalar, 3, 4>;
struct AffineBody
{
    mat3 A = mat3::Zero();
    vec3 p = vec3::Zero();
    q4 q0 = q4::Zero(), dqdt = q4::Zero();
    Eigen::MatrixXd deform(const Eigen::MatrixXd &V_rest)
    {
        return ((A * V_rest.transpose()).colwise() + p).transpose();
    }
};
struct GlobalVariables {
    int frame, max_frame = 100;
    vector<AffineBody> cubes;
    Eigen::MatrixXd V_rest;
    bool PA_STREAM_STARTED = false, terminated = false;
} globals;

struct PaModalData {
    std::unique_ptr<ModalSolver<double>> *solver;
    SoundMessage<double> soundMessage;
};
int PaModalCallback(const void *inputBuffer,
                    void *outputBuffer,
                    unsigned long framesPerBuffer,
                    const PaStreamCallbackTimeInfo* timeInfo,
                    PaStreamCallbackFlags statusFlags,
                    void *userData) {
    /* Cast data passed through stream to our structure. */
    PaModalData *data = (PaModalData*)userData;
    float *out = (float*)outputBuffer;
    unsigned int i;
    (void) inputBuffer; /* Prevent unused variable warning. */
    bool success = (*(data->solver))->dequeueSoundMessage(data->soundMessage);
    // VIEWER_SETTINGS.bufferHealth[VIEWER_SETTINGS.bufferHealthPtr] =
    //     (float)success;
    // VIEWER_SETTINGS.incrementBufferHealthPtr();
    for( i=0; i<framesPerBuffer; i++ ) {
        *out++ = (float)(data->soundMessage.data(i)/1E10);
        *out++ = (float)(data->soundMessage.data(i)/1E10);
    }
    return 0;
}

void player_load(
    const std::string &path,
    int timestep,
    vector<AffineBody> &cubes)
{
    string filename = path + "/" + to_string(timestep);
    ifstream in(filename, ios::in | ios::binary);
    int n_cubes = cubes.size();

    for (int i = 0; i < n_cubes; i++)
    {
        auto &c = cubes[i];
        in.read((char *)c.q0.data(), 12 * sizeof(scalar));
        in.read((char *)c.dqdt.data(), 12 * sizeof(scalar));
        c.p = c.q0.col(0);
        c.A = c.q0.block<3, 3>(0, 1);
    }
    in.close();
}

int main(int argc, char** argv) {


    // const string mat_file = "model_502009/502009_material.txt", mod_file = "model_502009/502009_eurf.modes", fat_path = "model_502009/502009_ffat_maps";
    const string mat_file = "model_00000/00000_material.txt", mod_file = "model_00000/00000_surf.modes", fat_path = "model_00000/00000_ffat_maps";
    int N_modesAudible;
    std::unique_ptr<ModalMaterial<double>> material(ReadMaterial<double>(
        mat_file.c_str()));
    std::unique_ptr<ModeData<double>> modes(
        ReadModes<double>(mod_file.c_str()));
    assert(modes->numDOF() == V1.rows()*3 && "DOFs mismatch");
    // build modal integrator and solver/scheduler
    std::unique_ptr<ModalSolver<double>> solver(
        BuildSolver(
            material,
            modes,
            fat_path,
            N_modesAudible
        )
    );
    // start a simulation thread and use max priority
    std::thread threadSim([&](){
        while (!globals.terminated) {
            solver->step();
        }
    });
    #ifdef _WIN32
    HANDLE h = (HANDLE)threadSim.native_handle();
    SetThreadPriority(h, THREAD_PRIORITY_TIME_CRITICAL);
    #else
    sched_param sch_params;
    sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(threadSim.native_handle(), SCHED_FIFO, &sch_params);
    #endif
    // setup audio callback stuff
    CHECK_PA_LAUNCH(Pa_Initialize());
    PaModalData paData;
    paData.solver = &solver;
    PaStream *stream;
    CHECK_PA_LAUNCH(Pa_OpenDefaultStream(&stream,
        0,                 /* no input channels */
        2,                 /* stereo output */
        paFloat32,         /* 32 bit floating point output */
        SAMPLE_RATE,       /* audio sampling rate */
        FRAMES_PER_BUFFER, /* frames per buffer */
        PaModalCallback,   /* audio callback function */
        &paData ));        /* audio callback data */

    globals.frame = 0;

    // create viewer
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.core().is_animating = true;
    unsigned main_view;
    int obj_id;
    viewer.callback_init = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
        // main viewport visualize the simulation sequence
        viewer.core().viewport = Eigen::Vector4f(0, 0, 1280, 800);
        main_view = viewer.core_list[0].id; 
        obj_id = viewer.data_list[0].id;
        return false;
    };
    viewer.callback_post_resize = [&](
            igl::opengl::glfw::Viewer &v, int w, int h) {
        v.core(main_view).viewport = Eigen::Vector4f(0, 0, w, h);
        return true;
    };
    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&]()
    {
        ImGui::SetNextWindowPos(ImVec2(0,0));
        ImGui::SetNextWindowSize(ImVec2(250,0));
        ImGui::Begin("Simulation");

        ImGui::Checkbox("Animate", &viewer.core().is_animating);
        ImGui::SliderInt("frame", &globals.frame, 0, globals.max_frame, "%d");
        ImGui::End();

    };

    Eigen::MatrixXd V, V1, VN; 
    Eigen::MatrixXi F, F1;
    igl::read_triangle_mesh("model_502009/flower.obj", V, F);

    igl::read_triangle_mesh("model_00000/00000.tet.obj", V1, F1);
    igl::per_vertex_normals(V1, F1, VN);

    globals.V_rest = V;
    globals.cubes.resize(1);
    viewer.data().set_mesh(V, F);

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer)->bool {
        // update simulation frame
        if (viewer.core().is_animating) {
            globals.frame = (globals.frame + 1) % globals.max_frame;
        }

        player_load("trace", globals.frame, globals.cubes);
        assert(globals.cubes.size() == 1);
        for (auto &c : globals.cubes)
        {
            V = c.deform(globals.V_rest);
            viewer.data(obj_id).set_vertices(V);
        }
        vec3 vn = VN.row(0).normalized();
        ForceMessage<double> force;
        GetModalForceVertex(N_modesAudible, *modes, 0, vn, force); 
        solver->enqueueForceMessageNoFail(force);
        return false;
    };
    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool
    {
        if (!globals.PA_STREAM_STARTED)
        {
            CHECK_PA_LAUNCH(Pa_StartStream(stream));
            globals.PA_STREAM_STARTED = true;
        }
        return false;
    };
    viewer.launch(true, false, "Simulation Viewer");
    CHECK_PA_LAUNCH(Pa_StopStream(stream));
    CHECK_PA_LAUNCH(Pa_CloseStream(stream));
    CHECK_PA_LAUNCH(Pa_Terminate());
    globals.terminated = true;
}