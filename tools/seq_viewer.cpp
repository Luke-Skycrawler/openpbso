#include <string>
#include <vector>
#include <memory>
#include "igl/opengl/glfw/Viewer.h"
#include "portaudio.h"
#include "modal_solver.h"
#include "forces.h"
#include "config.h"
#include "ModeData.h"
#include "ModalMaterial.h"
#include "igl/read_triangle_mesh.h"
#include "igl/per_vertex_normals.h"
#include "igl/opengl/glfw/imgui/ImGuiMenu.h"

/*------------------------------------
Creates a libigl viewer for simulation sequence with sound synthesis
*--------------------------------*/

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
} globals;

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
    // create viewer
    igl::opengl::glfw::Viewer viewer;
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


    igl::opengl::glfw::imgui::ImGuiMenu menu;
    globals.frame = 0;
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

    Eigen::MatrixXd V, VN; 
    Eigen::MatrixXi F;
    igl::read_triangle_mesh("model_502009/flower.obj", V, F);
    igl::per_vertex_normals(V, F, VN);
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
        return false;
    };
    viewer.launch(true, false, "Simulation Viewer");
    
}