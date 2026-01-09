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


struct GlobalVariables {
    int frame; 
} globals; 

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
        ImGui::SliderInt("frame", &globals.frame, 0, 1000, "%d");
        ImGui::End();
        
    };

    Eigen::MatrixXd V, VN; 
    Eigen::MatrixXi F;
    igl::read_triangle_mesh("model_00000/00000.tet.obj", V, F);
    igl::per_vertex_normals(V, F, VN);

    viewer.data().set_mesh(V, F);

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer)->bool {
        // update simulation frame
        if (viewer.core().is_animating) {
            globals.frame = (globals.frame + 1) % 1000;
        }

        return false;
    };
    viewer.launch(true, false, "Simulation Viewer");
    
}