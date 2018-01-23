//
// Created by yanhang on 3/19/17.
//

#include "main_widget.h"

namespace IMUProject {

namespace {
bool ReadResult(const std::string &path, const int frame_interval, std::vector<Eigen::Vector3d> *trajectory) {
  CHECK(trajectory);
  std::ifstream full_in(path.c_str());
  std::string line;
  if (full_in.is_open()) {
    printf("Loading %s\n", path.c_str());
    std::vector<Eigen::Vector3d> traj;
    std::getline(full_in, line);
    int count = 0;
    while (std::getline(full_in, line)) {
      std::vector<double> values = ParseCommaSeparatedLine(line);
      if (count % frame_interval == 0) {
        trajectory->emplace_back(values[2], values[3], values[4]);
      }
      count++;
    }
    return true;
  }else{
    return false;
  }
}
}  // namespace

MainWidget::MainWidget(const std::string &path,
                       const int canvas_width,
                       const int canvas_height,
                       QWidget *parent) : render_count_(0),
                                          panel_border_margin_(10), panel_size_(300), is_rendering_(false) {
  setFocusPolicy(Qt::StrongFocus);
  canvas_.reset(new Canvas(canvas_width, canvas_height));
  navigation_.reset(new Navigation(50.f,
                                   (float) width() / (float) height(),
                                   (float) canvas_width,
                                   (float) canvas_height));
//		QImage traj_legend_image("../../viewer/resource/images/traj_legend.png");
//		legend_areas_.emplace_back(width() / 2, 0.0, traj_legend_image.width(), traj_legend_image.height());
//		legends_.emplace_back(new LegendRenderer(traj_legend_image.width(), traj_legend_image.height(), traj_legend_image));

  InitializeTrajectories(path);
}

void MainWidget::InitializeTrajectories(const std::string &path) {
  double max_distance = -1.0;
  const double fill_ratio = 0.8;
  double ratio = -1;
  char buffer[128] = {};
  Eigen::Vector3d centroid(-1, -1, -1);

  std::vector<Eigen::Vector3f> traj_colors;

  const Eigen::Vector3f full_traj_color(0.0, 0.0, 0.8);
  const Eigen::Vector3f ori_traj_color(0.0, 0.8, 0.0);
  const Eigen::Vector3f mag_traj_color(0.5, 0, 0.5);
  const Eigen::Vector3f step_traj_color(0.31, 0.31, 0.31);
  const Eigen::Vector3f tango_traj_color(0.8, 0, 0);

  auto add_trajectory = [&](std::vector<Eigen::Vector3d> traj, std::vector<Eigen::Quaterniond> orientation,
                            const Eigen::Vector3f color, const float frustum_size,
                            Eigen::Quaterniond global_rotation) {
    CHECK_GT(traj.size(), 0);
    Eigen::Vector3d first_pos = traj[0];
    for (int i=0; i<traj.size(); ++i){
      traj[i] = global_rotation * (traj[i] - first_pos) + first_pos;
      orientation[i] = global_rotation * orientation[i];
    }
    if (max_distance < 0) {
      centroid = std::accumulate(traj.begin(), traj.end(), Eigen::Vector3d(0, 0, 0)) / (double) traj.size();
      double traj_max_distance = -1.0;
      for (auto i = 0; i < traj.size(); ++i) {
        double dis = (traj[i] - centroid).norm();
        traj_max_distance = std::max(traj_max_distance, dis);
      }
      max_distance = traj_max_distance;
      ratio = (double) std::min(canvas_->GetWidth() / 2, canvas_->GetHeight() / 2) / max_distance *
          fill_ratio;
    }

    for (auto& pos: traj){
      pos = (pos - centroid) * ratio;
    }

    trajectories_.emplace_back(new OfflineTrajectory(traj, color));
    view_frustum_.emplace_back(new ViewFrustum(frustum_size, true));
    positions_.push_back(traj);
    orientations_.push_back(orientation);
    traj_colors.push_back(color);
  };

  IMUDataset dataset(path);
  std::vector<Eigen::Quaterniond> gt_orientation;
  std::vector<Eigen::Quaterniond> imu_orientation;
  std::vector<Eigen::Vector3d> gt_position;
  ts_.clear();
  for (auto i = 0; i < dataset.GetTimeStamp().size(); i += frame_interval_) {
    ts_.push_back(dataset.GetTimeStamp()[i]);
    gt_orientation.push_back(dataset.GetOrientation()[i]);
    gt_position.push_back(dataset.GetPosition()[i]);
    imu_orientation.push_back(dataset.GetRotationVector()[i]);
  }

  Eigen::Matrix3d local_to_global;
  // local_to_global << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0;
  local_to_global = Eigen::Matrix3d::Identity();
  Eigen::Quaterniond init_orientation (local_to_global.inverse());
  Eigen::Vector3d sum_gt_position = std::accumulate(gt_position.begin(), gt_position.end(), Eigen::Vector3d(0, 0, 0));
  bool is_gt_valid = sum_gt_position.norm() > std::numeric_limits<double>::epsilon();

  const int start_portion_length = std::min(500, static_cast<int>(gt_position.size() - 1));
  Eigen::Quaterniond global_rotation = Eigen::Quaterniond::Identity();

  if (is_gt_valid){
    Eigen::Quaterniond imu_to_tango = gt_orientation[0] * imu_orientation[0].conjugate();
    Eigen::Vector3d init_offset = gt_position[start_portion_length] - gt_position[0];
    init_offset[2] = 0;
    global_rotation = Eigen::Quaterniond::FromTwoVectors(init_offset, Eigen::Vector3d(0, 1, 0));
    init_orientation = gt_orientation[0] * imu_orientation[0].conjugate();
    add_trajectory(gt_position, gt_orientation, tango_traj_color, 0.5f, global_rotation);
  } else {
    printf("Ground truth not presented\n");
  }
  
  for (auto &ori: imu_orientation) {
    ori = init_orientation * ori;
  }

//  {
//    sprintf(buffer, "%s/result_raw/result_raw.csv", path.c_str());
//    std::vector<Eigen::Vector3d> traj;
//    if (ReadResult(buffer, frame_interval_, &traj)){
//      if (!is_gt_valid){
//        Eigen::Vector3d init_offset = traj[start_portion_length] - traj[0];
//        init_offset[2] = 0;
//        global_rotation = Eigen::Quaterniond::FromTwoVectors(init_offset, Eigen::Vector3d(0, 1, 0));
//      }
//      add_trajectory(traj, imu_orientation, ori_traj_color, 1.0f, global_rotation);
//    }
//  }

  {
    sprintf(buffer, "%s/result_full/result_full.csv", path.c_str());
    std::vector<Eigen::Vector3d> traj;
    if (ReadResult(buffer, frame_interval_, &traj)){
      if (!is_gt_valid){
        Eigen::Vector3d init_offset = traj[start_portion_length] - traj[0];
        init_offset[2] = 0;
        global_rotation = Eigen::Quaterniond::FromTwoVectors(init_offset, Eigen::Vector3d(0, 1, 0));
      }
      add_trajectory(traj, imu_orientation, full_traj_color, 1.0f, global_rotation);
    }
  }

  {
    sprintf(buffer, "%s/result_step/result_step.csv", path.c_str());
    std::vector<Eigen::Vector3d> traj;
    if (ReadResult(buffer, frame_interval_, &traj)){
      add_trajectory(traj, imu_orientation, step_traj_color, 0.5f, global_rotation);
    }
  }

  {
    sprintf(buffer, "%s/result_ori_only/result_ori_only.csv", path.c_str());
    std::vector<Eigen::Vector3d> traj;
    if (ReadResult(buffer, frame_interval_, &traj)){
      add_trajectory(traj, imu_orientation, ori_traj_color, 0.5f, global_rotation);
    }
  }

  {
    sprintf(buffer, "%s/result_mag_only/result_mag_only.csv", path.c_str());
    std::vector<Eigen::Vector3d> traj;
    if (ReadResult(buffer, frame_interval_, &traj)){
      add_trajectory(traj, imu_orientation, mag_traj_color, 0.5f, global_rotation);
    }
  }


//  double traj_length = (gt_position.back() - gt_position[0]).norm();
//  std::vector<Eigen::Vector3d> y_pos_traj(imu_orientation.size(), Eigen::Vector3d(0, 0, 0));
//  for (int i=0; i<y_pos_traj.size(); ++i){
//    y_pos_traj[i][1] = traj_length / y_pos_traj.size() * i;
//  }
//  add_trajectory(y_pos_traj, imu_orientation, Eigen::Vector3f(0, 0, 0), 0.5, global_rotation);

  speed_panel_.reset(new OfflineSpeedPanel((int) traj_colors.size(), traj_colors, 1.0f, 1.5f * ratio));

  // Sanity checks
  CHECK_EQ(view_frustum_.size(), trajectories_.size());
  CHECK_EQ(view_frustum_.size(), positions_.size());
  CHECK_EQ(view_frustum_.size(), orientations_.size());
  for (auto i = 0; i < view_frustum_.size(); ++i) {
    CHECK_EQ(positions_[i].size(), ts_.size());
    CHECK_EQ(orientations_[i].size(), ts_.size());
  }
}

void MainWidget::initializeGL() {
  initializeOpenGLFunctions();

  canvas_->InitGL();
  for (auto i = 0; i < view_frustum_.size(); ++i) {
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
  for (auto i = 0; i < view_frustum_.size(); ++i) {
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

void MainWidget::UpdateCameraInfo(const int ind) {
  constexpr int ref_traj = 0;
  for (auto i = 0; i < view_frustum_.size(); ++i) {
    trajectories_[i]->SetRenderLength(ind);
    view_frustum_[i]->UpdateCameraPose(positions_[i][ind], orientations_[i][ind]);
  }

  std::vector<Eigen::Vector3d> speeds;
  if (ind > 0) {
    for (auto i = 0; i < view_frustum_.size(); ++i) {
      Eigen::Vector3d speed = (positions_[i][ind] - positions_[i][ind - 1])
          / (ts_[ind] - ts_[ind - 1]);
      speed[2] = 0.0;
      speeds.push_back(speed);
    }
    speed_panel_->UpdateDirections(speeds);
  }

  navigation_->UpdateNavitation(positions_[ref_traj][ind], orientations_[ref_traj][ind]);
}

void MainWidget::timerEvent(QTimerEvent *event) {
  if (render_count_ >= (int) ts_.size() - 1) {
    is_rendering_ = false;
  }
  UpdateCameraInfo(render_count_);
  update();
  if (is_rendering_) {
    render_count_++;
  }
}

void MainWidget::keyPressEvent(QKeyEvent *e) {
  switch (e->key()) {
    case (Qt::Key_1): {
      navigation_->SetCameraMode(PERSPECTIVE);
      UpdateCameraInfo(render_count_);
      update();
      break;
    }
    case (Qt::Key_2): {
      navigation_->SetCameraMode(SIDE);
      UpdateCameraInfo(render_count_);
      update();
      break;
    }
    case (Qt::Key_3): {
      navigation_->SetCameraMode(BACK);
      UpdateCameraInfo(render_count_);
      update();
      break;
    }
    case (Qt::Key_4): {
      navigation_->SetCameraMode(TOP);
      UpdateCameraInfo(render_count_);
      update();
      break;
    }
    case (Qt::Key_5): {
      navigation_->SetCameraMode(CENTER);
      UpdateCameraInfo(render_count_);
      update();
      break;
    }
    case (Qt::Key_S): {
      is_rendering_ = !is_rendering_;
      break;
    }
    case (Qt::Key_R): {
      render_count_ = 0;
      speed_panel_->ResetFilters();
      is_rendering_ = true;
    }
    case (Qt::Key_Left): {
      if (render_count_ > 10) {
        render_count_ -= 10;
        UpdateCameraInfo(render_count_);
        update();
      }
      break;
    }
    case (Qt::Key_Right): {
      render_count_ = render_count_ < ts_.size() - 10 ? render_count_ + 10 : 0;
      UpdateCameraInfo(render_count_);
      update();
      break;
    }
    case (Qt::Key_Up): {
      navigation_->IncreaseFOV();
      UpdateCameraInfo(render_count_);
      update();
      break;
    }
    case (Qt::Key_Down): {
      navigation_->DecreaseFOV();
      UpdateCameraInfo(render_count_);
      update();
      break;
    }
    default: break;
  }
}

}//namespace IMUProject
