//
// Created by yanhang on 3/19/17.
//

#include "main_widget.h"

using namespace std;

namespace IMUProject{

	MainWidget::MainWidget(const std::string &path,
                           const int canvas_width,
                           const int canvas_height,
                           QWidget *parent): render_count_(0),
	                                         full_traj_color(0.0f, 0.0f, 1.0f),
	                                         const_traj_color(0.6f, 0.6f, 0.0f),
	                                         tango_traj_color(1.0f, 0.0f, 0.0f),
	                                         panel_border_margin_(10), panel_size_(300), is_rendering_(false){
		setFocusPolicy(Qt::StrongFocus);
        canvas_.reset(new Canvas(canvas_width, canvas_height));
		navigation_.reset(new Navigation(50.f, (float)width() / (float)height(), (float)canvas_width, (float)canvas_height));
        speed_panel_.reset(new OfflineSpeedPanel());

//		QImage traj_legend_image("../../viewer/resource/images/traj_legend.png");
//		legend_areas_.emplace_back(width() / 2, 0.0, traj_legend_image.width(), traj_legend_image.height());
//		legends_.emplace_back(new LegendRenderer(traj_legend_image.width(), traj_legend_image.height(), traj_legend_image));

		InitializeTrajectories(path);
	}

	void MainWidget::InitializeTrajectories(const std::string& path) {
		double max_distance = -1.0;
		const double fill_ratio = 0.8;
		double ratio = -1;
		char buffer[128] = {};
		Eigen::Vector4d bbox(-1, -1, -1, -1);
		Eigen::Vector3d centroid(-1, -1, -1);

		auto add_trajectory = [&](std::vector<Eigen::Vector3d>& traj, const std::vector<Eigen::Quaterniond>& orientation,
		                          const Eigen::Vector3f color, const float frustum_size){
			CHECK_GT(traj.size(), 0);
			if(max_distance < 0) {
//				for (auto i = 0; i < traj.size(); ++i) {
//					if (bbox[0] < 0 || traj[i][0] < bbox[0]) {
//						bbox[0] = traj[i][0];
//					}
//					if (bbox[1] < 0 || traj[i][0] > bbox[0]) {
//						bbox[1] = traj[i][0];
//					}
//					if (bbox[2] < 0 || traj[i][1] < bbox[2]) {
//						bbox[2] = traj[i][1];
//					}
//					if (bbox[3] < 0 || traj[i][1] > bbox[3]) {
//						bbox[3] = traj[i][1];
//					}
//				}
//                centroid = Eigen::Vector3d((bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2, 1.0);
                centroid = std::accumulate(traj.begin(), traj.end(), Eigen::Vector3d(0, 0, 0)) / (double)traj.size();
				double traj_max_distance = -1.0;
				for (auto i = 0; i < traj.size(); ++i) {
					double dis = (traj[i] - centroid).norm();
					traj_max_distance = std::max(traj_max_distance, dis);
				}
				max_distance = traj_max_distance;
				ratio = (double) std::min(canvas_->GetWidth() / 2, canvas_->GetHeight() / 2) / max_distance *
				        fill_ratio;
//                ratio = 1.0;
			}

			for(auto& pos: traj){
				pos = (pos - centroid) * ratio;
			}
			trajectories_.emplace_back(new OfflineTrajectory(traj, color));
			view_frustum_.emplace_back(new ViewFrustum(frustum_size, true));
			positions_.push_back(traj);
			orientations_.push_back(orientation);
		};

		IMUDataset dataset(path);
		std::vector<Eigen::Quaterniond> gt_orientation;
		std::vector<Eigen::Quaterniond> imu_orientation;
		std::vector<Eigen::Vector3d> gt_position;
		ts_.clear();
		for(auto i=0; i<dataset.GetTimeStamp().size(); i+=frame_interval_){
			ts_.push_back(dataset.GetTimeStamp()[i]);
			gt_orientation.push_back(dataset.GetOrientation()[i]);
			gt_position.push_back(dataset.GetPosition()[i]);
			imu_orientation.push_back(dataset.GetRotationVector()[i]);
		}

		Eigen::Quaterniond imu_to_tango = gt_orientation[0] * imu_orientation[0].conjugate();
		for(auto &ori: imu_orientation){
			ori = imu_to_tango * ori;
		}

		sprintf(buffer, "%s/result_full.csv", path.c_str());
		string line;
		ifstream full_in(buffer);
		if(full_in.is_open()){
			std::vector<Eigen::Vector3d> traj;
			std::getline(full_in, line);
			int count = 0;
			while(std::getline(full_in, line)) {
				std::vector<double> values = ParseCommaSeparatedLine(line);
				if(count %  frame_interval_ == 0) {
					traj.emplace_back(values[2], values[3], values[4]);
				}
				count++;
			}
			add_trajectory(traj, imu_orientation, full_traj_color, 1.0f);
		}

		sprintf(buffer, "%s/result_const.csv", path.c_str());
		ifstream const_in(buffer);
		if(const_in.is_open()){
			std::vector<Eigen::Vector3d> traj;
			std::getline(const_in, line);
			int count = 0;
			while(std::getline(const_in, line)) {
				std::vector<double> values = ParseCommaSeparatedLine(line);
				if(count %  frame_interval_ == 0) {
					traj.emplace_back(values[2], values[3], values[4]);
				}
				count++;
			}
			add_trajectory(traj, imu_orientation, const_traj_color, 0.5f);
		}

		add_trajectory(gt_position, gt_orientation, tango_traj_color, 0.5f);

        // Sanity checks
		CHECK_EQ(view_frustum_.size(), trajectories_.size());
		CHECK_EQ(view_frustum_.size(), positions_.size());
		CHECK_EQ(view_frustum_.size(), orientations_.size());
		for(auto i=0; i<view_frustum_.size(); ++i){
			CHECK_EQ(positions_[i].size(), ts_.size());
			CHECK_EQ(orientations_[i].size(), ts_.size());
		}
	}

	void MainWidget::initializeGL() {
		initializeOpenGLFunctions();

		canvas_->InitGL();
		for(auto i=0; i<view_frustum_.size(); ++i){
			trajectories_[i]->InitGL();
			view_frustum_[i]->InitGL();
		}
        speed_panel_->InitGL();
//		for(auto& v: legends_){
//			v->InitGL();
//		}

        UpdateCameraInfo(0);

		glClearColor(1.f, 1.f, 1.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);
		glViewport(0, 0, width(), height());

        render_timer_.start(frame_interval_ * 5, this);
	}

	void MainWidget::resizeGL(int w, int h) {
        glViewport(0, 0, w, h);
	}

	void MainWidget::paintGL() {
		canvas_->Render(*navigation_);
		for(auto i=0; i<1; ++i){
			trajectories_[i]->Render(*navigation_);
			view_frustum_[i]->Render(*navigation_);
		}

        // Render the speed panel
        glViewport(panel_border_margin_, panel_border_margin_, width() / 6, width() / 6);
        speed_panel_->Render(*navigation_);

//		for(int i=0; i<legends_.size(); ++i){
//			glViewport(legend_areas_[i].x(), legend_areas_[i].y(), legend_areas_[i].width(),
//			           legend_areas_[i].height());
//			legends_[i]->Render(*navigation_);
//		}

		glViewport(0, 0, width(), height());
		glFlush();
	}

    void MainWidget::UpdateCameraInfo(const int ind){
	    constexpr int ref_traj = 0;
	    for(auto i=0; i<view_frustum_.size(); ++i) {
		    trajectories_[i]->SetRenderLength(ind);
		    view_frustum_[i]->UpdateCameraPose(positions_[i][ind], orientations_[i][ind]);
	    }
        if(ind > 0){
            Eigen::Vector3d forward_dir = (positions_[ref_traj][ind] - positions_[ref_traj][ind - 1])
                                          / (ts_[ind] - ts_[ind-1]);
            forward_dir[2] = 0.0;
            Eigen::Vector3d device_dir = orientations_[ref_traj][ind] * Eigen::Vector3d(0, 0, -1);
            device_dir[2] = 0.0;
            speed_panel_->UpdateDirection(forward_dir, device_dir);
        }else{
            speed_panel_->UpdateDirection(Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 1, 0));
        }

        navigation_->UpdateNavitation(positions_[ref_traj][ind], orientations_[ref_traj][ind]);
    }

	void MainWidget::timerEvent(QTimerEvent *event) {
        if(render_count_ >= (int)ts_.size() - 1){
            is_rendering_ = false;
        }
        UpdateCameraInfo(render_count_);
        update();
        if(is_rendering_) {
            render_count_++;
        }
	}

	void MainWidget::keyPressEvent(QKeyEvent *e) {
		switch(e->key()){
            case(Qt::Key_1):{
                navigation_->SetCameraMode(PERSPECTIVE);
                UpdateCameraInfo(render_count_);
                update();
                break;
            }
            case(Qt::Key_2):{
                navigation_->SetCameraMode(CENTER);
                UpdateCameraInfo(render_count_);
                update();
                break;
            }
            case(Qt::Key_3):{
                navigation_->SetCameraMode(BACK);
                UpdateCameraInfo(render_count_);
                update();
                break;
            }
            case(Qt::Key_4):{
                navigation_->SetCameraMode(TOP);
                UpdateCameraInfo(render_count_);
                update();
                break;
            }
			case(Qt::Key_S):{
                is_rendering_ = !is_rendering_;
				break;
			}
            case(Qt::Key_R):{
                render_count_ = 0;
                is_rendering_ = true;
            }
			default:
				break;
		}
	}

}//namespace IMUProject