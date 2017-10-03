//
// Created by yanhang on 3/5/17.
//

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <gflags/gflags.h>

#include <utility/data_io.h>
#include <utility/utility.h>

#include "imu_localization.h"

DEFINE_string(model_path, "../../../../models/model_0326_full_w200_s20", "Path to model");
DEFINE_string(mapinfo_path, "default", "path to map info");
DEFINE_int32(log_interval, 1000, "logging interval");
DEFINE_string(color, "blue", "color");
DEFINE_double(weight_vs, 1.0, "weight_vs");
DEFINE_double(weight_ls, 1.0, "weight_ls");
DEFINE_string(id, "full", "suffix");
DEFINE_string(preset, "none", "preset mode");

DEFINE_bool(run_global, true, "Run global optimization at the end");
DEFINE_bool(tango_ori, false, "Use ground truth orientation");

using namespace std;

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "Usage: ./IMULocalization_cli <path-to-data>" << endl;
    return 1;
  }

  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = true;

  char buffer[256] = {};

  LOG(INFO) << "Initializing...";
  // load data
  IMUProject::IMUDataset dataset(argv[1]);

  // load regressor
  std::vector<cv::Ptr<cv::ml::SVM> > regressors(3);
  for (int chn: {0, 2}) {
    sprintf(buffer, "%s_%d.yml", FLAGS_model_path.c_str(), chn);
    regressors[chn] = cv::ml::SVM::load(std::string(buffer));
    LOG(INFO) << buffer << " loaded";
  }

  // Run the system
  const int N = (int) dataset.GetTimeStamp().size();
  const std::vector<double> &ts = dataset.GetTimeStamp();
  const std::vector<Eigen::Vector3d> &gyro = dataset.GetGyro();
  const std::vector<Eigen::Vector3d> &linacce = dataset.GetLinearAcceleration();
  const std::vector<Eigen::Vector3d> &gravity = dataset.GetGravity();

  std::vector<Eigen::Quaterniond> orientation;
  if (FLAGS_tango_ori) {
    orientation = dataset.GetOrientation();
  } else {
    orientation = dataset.GetRotationVector();
    Eigen::Quaterniond rv_to_tango = dataset.GetOrientation()[0] * dataset.GetRotationVector()[0].conjugate();
    for (auto &v: orientation) {
      v = rv_to_tango * v;
    }
  }

  Eigen::Vector3d traj_color(0, 0, 255);
  if (FLAGS_color == "yellow") {
    traj_color = Eigen::Vector3d(128, 128, 0);
  } else if (FLAGS_color == "green") {
    traj_color = Eigen::Vector3d(0, 128, 0);
  } else if (FLAGS_color == "brown") {
    traj_color = Eigen::Vector3d(0, 128, 128);
  }

  // crete trajectory instance
  IMUProject::IMULocalizationOption option;
  const double sigma = 0.2;
  option.weight_ls_ = FLAGS_weight_ls;
  option.weight_vs_ = FLAGS_weight_vs;
  if (FLAGS_preset == "full") {
    option.reg_option_ = IMUProject::FULL;
    FLAGS_id = "full";
    traj_color = Eigen::Vector3d(0, 0, 255);
  } else if (FLAGS_preset == "ori_only") {
    option.reg_option_ = IMUProject::ORI;
    FLAGS_id = "ori_only";
    traj_color = Eigen::Vector3d(0, 200, 0);
  } else if (FLAGS_preset == "mag_only") {
    option.reg_option_ = IMUProject::MAG;
    FLAGS_id = "mag_only";
    traj_color = Eigen::Vector3d(139, 0, 139);
  } else if (FLAGS_preset == "const") {
    option.reg_option_ = IMUProject::CONST;
    FLAGS_id = "const";
    traj_color = Eigen::Vector3d(150, 150, 0);
  }

  IMUProject::IMUTrajectory trajectory(Eigen::Vector3d(0, 0, 0), dataset.GetPosition()[0], regressors, sigma, option);

  float start_t = (float) cv::getTickCount();

  constexpr int init_capacity = 20000;
  std::vector<Eigen::Vector3d> positions_opt;
  std::vector<Eigen::Quaterniond> orientations_opt;
//    positions_opt.reserve(init_capacity);
//    orientations_opt.reserve(init_capacity);

  for (int i = 0; i < N; ++i) {
    trajectory.AddRecord(ts[i], gyro[i], linacce[i], gravity[i], orientation[i]);

    if (i > option.local_opt_window_) {
      if (i % option.global_opt_interval_ == 0) {
//                LOG(INFO) << "Running global optimzation at frame " << i;
////                trajectory.RunOptimization(0, trajectory.GetNumFrames());
//
//                //block the execution is there are too many tasks in the background thread
//                while(true) {
//                    if(trajectory.CanAdd()){
//                        break;
//                    }
//                }
//                trajectory.ScheduleOptimization(0, trajectory.GetNumFrames());
      } else if (i % option.local_opt_interval_ == 0) {
        LOG(INFO) << "Running local optimzation at frame " << i;
        while (true) {
          if (trajectory.CanAdd()) {
            break;
          }
        }
        trajectory.ScheduleOptimization(i - option.local_opt_window_, option.local_opt_window_);
//	            trajectory.RunOptimization(i  - option.local_opt_window_, option.local_opt_window_);
      }
    }
    if (FLAGS_log_interval > 0 && i > 0 && i % FLAGS_log_interval == 0) {
      const float time_passage = std::max(((float) cv::getTickCount() - start_t) / (float) cv::getTickFrequency(),
                                          std::numeric_limits<float>::epsilon());
      sprintf(buffer, "%d records added in %.5fs, fps=%.2fHz\n", i, time_passage, (float) i / time_passage);
      LOG(INFO) << buffer;
    }

  }

  trajectory.EndTrajectory();
  if (FLAGS_run_global) {
    printf("Running global optimization on the whole sequence...\n");
    trajectory.RunOptimization(0, trajectory.GetNumFrames());
  }

  printf("All done. Number of points on trajectory: %d\n", trajectory.GetNumFrames());
  const float fps_all =
      (float) trajectory.GetNumFrames() / (((float) cv::getTickCount() - start_t) / (float) cv::getTickFrequency());
  printf("Overall framerate: %.3f\n", fps_all);

  sprintf(buffer, "%s/result_trajectory_%s.ply", argv[1], FLAGS_id.c_str());
  IMUProject::WriteToPly(std::string(buffer), dataset.GetTimeStamp().data(), trajectory.GetPositions().data(),
                         trajectory.GetOrientations().data(), trajectory.GetNumFrames(),
                         true, traj_color, 0.8, 100, 300);

  sprintf(buffer, "%s/tango_trajectory.ply", argv[1]);
  IMUProject::WriteToPly(std::string(buffer), dataset.GetTimeStamp().data(), dataset.GetPosition().data(),
                         dataset.GetOrientation().data(), (int) dataset.GetPosition().size(),
                         true, Eigen::Vector3d(255, 0, 0), 0.8, 100, 300);

  {
    // Write trajectory with double integration
    vector<Eigen::Vector3d> raw_traj(dataset.GetTimeStamp().size(), dataset.GetPosition()[0]);
    vector<Eigen::Vector3d> raw_speed(dataset.GetTimeStamp().size(), Eigen::Vector3d(0, 0, 0));
    for (auto i = 1; i < raw_traj.size(); ++i) {
      Eigen::Vector3d acce = orientation[i - 1] * dataset.GetLinearAcceleration()[i - 1];
      raw_speed[i] = raw_speed[i - 1] + acce * (ts[i] - ts[i - 1]);
      raw_traj[i] = raw_traj[i - 1] + raw_speed[i - 1] * (ts[i] - ts[i - 1]);
    }
    sprintf(buffer, "%s/raw.ply", argv[1]);
    IMUProject::WriteToPly(std::string(buffer), ts.data(), raw_traj.data(), orientation.data(),
                           (int) raw_traj.size(), true, Eigen::Vector3d(0, 128, 128));

    sprintf(buffer, "%s/result_raw.csv", argv[1]);
    ofstream raw_out(buffer);
    CHECK(raw_out.is_open());
    raw_out << ",time,pos_x,pos_y,pos_z,speed_x,speed_y,speed_z,bias_x,bias_y,bias_z" << endl;
    for (auto i = 0; i < raw_traj.size(); ++i) {
      const Eigen::Vector3d &pos = raw_traj[i];
      const Eigen::Vector3d &acce = dataset.GetLinearAcceleration()[i];
      const Eigen::Vector3d &spd = raw_speed[i];
      sprintf(buffer, "%d,%.9f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
              i, dataset.GetTimeStamp()[i], pos[0], pos[1], pos[2], spd[0], spd[1], spd[2],
              acce[0] - linacce[i][0], acce[1] - linacce[i][1], acce[2] - linacce[i][2]);
      raw_out << buffer;
    }
  }

  {
    // Write the trajectory and bias as txt
    sprintf(buffer, "%s/result_%s.csv", argv[1], FLAGS_id.c_str());
    ofstream traj_out(buffer);
    CHECK(traj_out.is_open());
    traj_out << ",time,pos_x,pos_y,pos_z,speed_x,speed_y,speed_z,bias_x,bias_y,bias_z" << endl;
    for (auto i = 0; i < trajectory.GetNumFrames(); ++i) {
      const Eigen::Vector3d &pos = trajectory.GetPositions()[i];
      const Eigen::Vector3d &acce = trajectory.GetLinearAcceleration()[i];
      const Eigen::Vector3d &spd = trajectory.GetSpeed()[i];
      sprintf(buffer, "%d,%.9f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
              i, dataset.GetTimeStamp()[i], pos[0], pos[1], pos[2], spd[0], spd[1], spd[2],
              acce[0] - linacce[i][0], acce[1] - linacce[i][1], acce[2] - linacce[i][2]);
      traj_out << buffer;
    }

    sprintf(buffer, "%s/regression_%s.txt", argv[1], FLAGS_id.c_str());
    ofstream reg_out(buffer);
    const std::vector<int> &cids = trajectory.GetConstraintInd();
    const std::vector<Eigen::Vector3d> &lss = trajectory.GetLocalSpeed();
    for (auto i = 0; i < cids.size(); ++i) {
      reg_out << cids[i] << ' ' << lss[i][0] << ' ' << lss[i][1] << ' ' << lss[i][2] << endl;
    }
  }

  if (FLAGS_mapinfo_path == "default") {
    sprintf(buffer, "%s/map.txt", argv[1]);
  } else {
    sprintf(buffer, "%s/%s", argv[1], FLAGS_mapinfo_path.c_str());
  }
  ifstream map_in(buffer);
  if (map_in.is_open()) {
    LOG(INFO) << "Found map info file, creating overlay";
    string line;
    map_in >> line;
    sprintf(buffer, "%s/%s", argv[1], line.c_str());

    cv::Mat map_img = cv::imread(buffer);
    CHECK(map_img.data) << "Can not open image file: " << buffer;

    Eigen::Vector2d sp1, sp2;
    Eigen::Vector3d op1(0, 0, 0), op2(0, 0, 0);
    double scale_length;

    map_in >> sp1[0] >> sp1[1] >> sp2[0] >> sp2[1];
    map_in >> scale_length;
    map_in >> op1[0] >> op1[1] >> op2[0] >> op2[1];

    Eigen::Vector2d start_pix(op1[0], op1[1]);

    const double pixel_length = scale_length / (sp2 - sp1).norm();

    IMUProject::TrajectoryOverlay(pixel_length, start_pix, op2 - op1, trajectory.GetPositions(),
                                  Eigen::Vector3d(255, 0, 0), map_img);

    IMUProject::TrajectoryOverlay(pixel_length, start_pix, op2 - op1, dataset.GetPosition(),
                                  Eigen::Vector3d(0, 0, 255), map_img);
    sprintf(buffer, "%s/overlay_%s.png", argv[1], FLAGS_id.c_str());
    cv::imwrite(buffer, map_img);
  }

  return 0;
}