//
// Created by yanhang on 3/19/17.
//

#include "renderable.h"

#include <glog/logging.h>

namespace IMUProject {

////////////////////////////////////
// Implementation of canvas
Canvas::Canvas(const float width, const float height,
               const float grid_size, const Eigen::Vector3f grid_color, const cv::Mat *texture)
    : width_(width), height_(height), line_alpha_(0.3f) {
  is_shader_init_ = false;
  vertex_data_ = {-width / 2.0f, 0.0f, -height / 2.0f,
                  width / 2.0f, 0.0f, -height / 2.0f,
                  width / 2.0f, 0.0f, height / 2.0f,
                  -width / 2.0f, 0.0f, height / 2.0f};
  index_data_ = {0, 1, 2, 2, 3, 0};
  texcoord_data_ = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};

  // Add grid lines
  GLuint p_counter = 0;
  for (float x = -width / 2.0f + grid_size; x <= width / 2.0f - grid_size; x += grid_size) {
    grid_vertex_data_.push_back(x);
    grid_vertex_data_.push_back(0.0f);
    grid_vertex_data_.push_back(-height_ / 2.0f);
    grid_vertex_data_.push_back(x);
    grid_vertex_data_.push_back(0.0f);
    grid_vertex_data_.push_back(height_ / 2.0f);

    grid_color_data_.push_back(grid_color[0]);
    grid_color_data_.push_back(grid_color[1]);
    grid_color_data_.push_back(grid_color[2]);
    grid_color_data_.push_back(line_alpha_);
    grid_color_data_.push_back(grid_color[0]);
    grid_color_data_.push_back(grid_color[1]);
    grid_color_data_.push_back(grid_color[2]);
    grid_color_data_.push_back(line_alpha_);

    grid_index_data_.push_back(p_counter);
    grid_index_data_.push_back(p_counter + 1);
    p_counter += (GLuint) 2;
  }
  for (float y = -height / 2.0f + grid_size; y < height / 2.0f; y += grid_size) {
    grid_vertex_data_.push_back(width / 2.0f);
    grid_vertex_data_.push_back(0.0f);
    grid_vertex_data_.push_back(y);
    grid_vertex_data_.push_back(-width / 2.0f);
    grid_vertex_data_.push_back(0.0f);
    grid_vertex_data_.push_back(y);

    //printf("(%.6f,%.6f,%.6f), (%.6f,%.6f,%.6f)\n", -width/2.0f, 0.0f, y, width/2.0f, 0.0f, y);
    grid_color_data_.push_back(grid_color[0]);
    grid_color_data_.push_back(grid_color[1]);
    grid_color_data_.push_back(grid_color[2]);
    grid_color_data_.push_back(line_alpha_);
    grid_color_data_.push_back(grid_color[0]);
    grid_color_data_.push_back(grid_color[1]);
    grid_color_data_.push_back(grid_color[2]);
    grid_color_data_.push_back(line_alpha_);

    grid_index_data_.push_back(p_counter);
    grid_index_data_.push_back(p_counter + 1);
    p_counter += (GLuint) 2;
  }

  if (texture == nullptr) {
    texture_img_.load(":/images/ground_texture.png");
    printf("Default texture image loaded. Width: %d, height: %d\n",
           texture_img_.width(), texture_img_.height());
  } else {
    QImage::Format tex_format = QImage::Format_RGB888;
    CHECK(texture->data) << "Empty texture image";
    if (texture->type() == CV_8UC3) {
      tex_format = QImage::Format_RGB888;
    } else if (texture->type() == CV_8UC4) {
      tex_format = QImage::Format_RGBA8888;
    } else {
      CHECK(true) << "Unsupported pixel format:" << texture->type();
    }

    texture_img_ = QImage(texture->data, texture->cols, texture->rows, tex_format);
  }

}

Canvas::~Canvas() {
  glDeleteBuffers(1, &vertex_buffer_);
  glDeleteBuffers(1, &index_buffer_);
  glDeleteBuffers(1, &texcoord_buffer_);
  glDeleteBuffers(1, &grid_vertex_buffer_);
  glDeleteBuffers(1, &grid_color_buffer_);
  glDeleteBuffers(1, &grid_index_buffer_);
}

void Canvas::InitGL() {
  initializeOpenGLFunctions();
  tex_shader_.reset(new QOpenGLShaderProgram());
  CHECK(tex_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/canvas_shader.vert"));
  CHECK(tex_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/canvas_shader.frag"));
  CHECK(tex_shader_->link()) << "Canvas: can not link texture shader";
  CHECK(tex_shader_->bind()) << "Canvas: can not bind texture shader";

  tex_shader_->enableAttributeArray("pos");
  tex_shader_->enableAttributeArray("texcoord");
  tex_shader_->release();

  line_shader_.reset(new QOpenGLShaderProgram());
  CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/line_shader.vert"));
  CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/line_shader.frag"));
  CHECK(line_shader_->link()) << "Canvas: can not link line shader";
  CHECK(line_shader_->bind());
  line_shader_->enableAttributeArray("pos");
  line_shader_->enableAttributeArray("v_color");
  line_shader_->release();

  is_shader_init_ = true;

  glGenBuffers(1, &vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, vertex_data_.size() * sizeof(GLfloat),
               vertex_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &texcoord_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
  glBufferData(GL_ARRAY_BUFFER, texcoord_data_.size() * sizeof(GLfloat),
               texcoord_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &index_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data_.size() * sizeof(GLuint),
               index_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &grid_vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, grid_vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, grid_vertex_data_.size() * sizeof(GLfloat),
               grid_vertex_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &grid_color_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, grid_color_buffer_);
  glBufferData(GL_ARRAY_BUFFER, grid_color_data_.size() * sizeof(GLfloat),
               grid_color_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &grid_index_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, grid_index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, grid_index_data_.size() * sizeof(GLuint),
               grid_index_data_.data(), GL_STATIC_DRAW);

  glEnable(GL_TEXTURE_2D);
  canvas_texture_.reset(new QOpenGLTexture(texture_img_));
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
}

void Canvas::Render(const Navigation &navigation) {
  CHECK(tex_shader_->bind());

  glEnable(GL_TEXTURE_2D);
  canvas_texture_->bind();
  CHECK(canvas_texture_->isBound()) << "Can not bind canvas texture";
  tex_shader_->setUniformValue("tex_sampler", 0);
  tex_shader_->setUniformValue("m_mat", navigation.GetModelViewMatrix());
  tex_shader_->setUniformValue("p_mat", navigation.GetProjectionMatrix());

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  tex_shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3);
  glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
  tex_shader_->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glDrawElements(GL_TRIANGLES, (GLsizei) index_data_.size(), GL_UNSIGNED_INT, 0);
  glDisable(GL_TEXTURE_2D);
  tex_shader_->release();

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  CHECK(line_shader_->bind());
  glLineWidth(1.0f);
  line_shader_->setUniformValue("m_mat", navigation.GetModelViewMatrix());
  line_shader_->setUniformValue("p_mat", navigation.GetProjectionMatrix());
  glBindBuffer(GL_ARRAY_BUFFER, grid_vertex_buffer_);
  line_shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3);
  glBindBuffer(GL_ARRAY_BUFFER, grid_color_buffer_);
  line_shader_->setAttributeBuffer("v_color", GL_FLOAT, 0, 4);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, grid_index_buffer_);
  glDrawElements(GL_LINES, (GLsizei) grid_index_data_.size(), GL_UNSIGNED_INT, 0);
  line_shader_->release();
  glDisable(GL_BLEND);
}

///////////////////////////////////
// Implementation of ViewFrustum
ViewFrustum::ViewFrustum(const float length, const bool with_axes, const float default_height)
    : default_height_(default_height) {
  vertex_data_ = {0.0f, 0.0f, 0.0f,
                  -length / 2.0f, length / 2.0f, -length * 0.8f,
                  length / 2.0f, length / 2.0f, -length * 0.8f,
                  length / 2.0f, -length / 2.0f, -length * 0.8f,
                  -length / 2.0f, -length / 2.0f, -length * 0.8f};
  index_data_ = {0, 1, 0, 2, 0, 3, 0, 4,
                 1, 2, 2, 3, 3, 4, 4, 1};
  color_data_ = {0.0f, 0.0f, 0.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 1.0f};

  if (with_axes) {
    vertex_data_.insert(vertex_data_.end(), {0.0f, 0.0f, 0.0f,
                                             length, 0.0f, 0.0f,
                                             0.0f, 0.0f, 0.0f,
                                             0.0f, length, 0.0f,
                                             0.0f, 0.0f, 0.0f,
                                             0.0f, 0.0f, length});
    index_data_.insert(index_data_.end(), {5, 6, 7, 8, 9, 10});
    color_data_.insert(color_data_.end(), {1.0f, 0.0f, 0.0f, 1.0f,
                                           1.0f, 0.0f, 0.0f, 1.0f,
                                           0.0f, 1.0f, 0.0f, 1.0f,
                                           0.0f, 1.0f, 0.0f, 1.0f,
                                           0.0f, 0.0f, 1.0f, 1.0f,
                                           0.0f, 0.0f, 1.0f, 1.0f});
  }

  float mat3[9] = {1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0,
                   0.0, -1.0f, 0.0};
  QMatrix3x3 l_to_g(mat3);
  local_to_global_ = QQuaternion::fromRotationMatrix(l_to_g);

}

void ViewFrustum::InitGL() {
  initializeOpenGLFunctions();

  line_shader_.reset(new QOpenGLShaderProgram());
  CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/line_shader.vert"));
  CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/line_shader.frag"));
  CHECK(line_shader_->link()) << "ViewFrustum: can not link line shader";
  CHECK(line_shader_->bind());
  line_shader_->enableAttributeArray("pos");
  line_shader_->enableAttributeArray("v_color");
  line_shader_->release();

  is_shader_init_ = true;

  glGenBuffers(1, &vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, vertex_data_.size() * sizeof(GLfloat),
               vertex_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &color_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
  glBufferData(GL_ARRAY_BUFFER, color_data_.size() * sizeof(GLfloat),
               color_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &index_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data_.size() * sizeof(GLuint),
               index_data_.data(), GL_STATIC_DRAW);
}

void ViewFrustum::Render(const Navigation &navigation) {
  CHECK(line_shader_->bind());

  QMatrix4x4 modelview;
  modelview.setToIdentity();
  modelview.translate(position_);
  modelview.rotate(local_to_global_);
  modelview.rotate(orientation_);
  modelview = navigation.GetModelViewMatrix() * modelview;

  line_shader_->setUniformValue("m_mat", modelview);
  line_shader_->setUniformValue("p_mat", navigation.GetProjectionMatrix());

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  line_shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3);
  glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
  line_shader_->setAttributeBuffer("v_color", GL_FLOAT, 0, 4);

  glLineWidth(2.0f);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glDrawElements(GL_LINES, (GLsizei) index_data_.size(), GL_UNSIGNED_INT, 0);
  line_shader_->release();
}

///////////////////////////////////
// Implementation of OfflineTrajectory
OfflineTrajectory::OfflineTrajectory(const std::vector<Eigen::Vector3d> &trajectory,
                                     const Eigen::Vector3f &color,
                                     const float default_height) {
  is_shader_init_ = false;
  vertex_data_.resize(trajectory.size() * 3);
  color_data_.resize(trajectory.size() * 4);
  index_data_.resize(trajectory.size());

  for (auto i = 0; i < trajectory.size(); ++i) {
    vertex_data_[3 * i] = (GLfloat) trajectory[i][0];
    vertex_data_[3 * i + 1] = (GLfloat) trajectory[i][2] + default_height;
    vertex_data_[3 * i + 2] = -1 * (GLfloat) trajectory[i][1];

    color_data_[4 * i] = color[0];
    color_data_[4 * i + 1] = color[1];
    color_data_[4 * i + 2] = color[2];
    color_data_[4 * i + 3] = 1.0f;

    index_data_[i] = (GLuint) i;
  }

}

void OfflineTrajectory::InitGL() {
  initializeOpenGLFunctions();

  line_shader_.reset(new QOpenGLShaderProgram());
  CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/line_shader.vert"));
  CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/line_shader.frag"));
  CHECK(line_shader_->link()) << "OfflineTrajectory: can not link line shader";
  CHECK(line_shader_->bind());
  line_shader_->enableAttributeArray("pos");
  line_shader_->enableAttributeArray("v_color");
  line_shader_->release();

  is_shader_init_ = true;

  glGenBuffers(1, &vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, vertex_data_.size() * sizeof(GLfloat),
               vertex_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &color_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
  glBufferData(GL_ARRAY_BUFFER, color_data_.size() * sizeof(GLfloat),
               color_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &index_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data_.size() * sizeof(GLuint),
               index_data_.data(), GL_STATIC_DRAW);
}

void OfflineTrajectory::Render(const Navigation &navigation) {
  CHECK(line_shader_->bind());
  line_shader_->setUniformValue("m_mat", navigation.GetModelViewMatrix());
  line_shader_->setUniformValue("p_mat", navigation.GetProjectionMatrix());

  glLineWidth(2.0f);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  line_shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3);

  glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
  line_shader_->setAttributeBuffer("v_color", GL_FLOAT, 0, 4);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glDrawElements(GL_LINE_STRIP, (GLsizei) render_length_, GL_UNSIGNED_INT, 0);

  line_shader_->release();
}

OfflineSpeedPanel::OfflineSpeedPanel(const int kTraj,
                                     const std::vector<Eigen::Vector3f> &colors,
                                     const float radius,
                                     const float max_speed,
                                     const QVector3D initial_dir)
    : kTraj_(kTraj), initial_dir_(initial_dir), filter_alpha_(0.8f), radius_(radius),
      max_speed_(max_speed), panel_alpha_(0.5f), z_pos_(-15.0f) {
  CHECK_EQ(colors.size(), kTraj);
  const float circle_divide = 200;
  panel_vertex_data_.reserve((int) circle_divide * 3 + 3);
  panel_color_data_.reserve((int) circle_divide * 4 + 4);
  panel_index_data_.reserve((int) circle_divide + 1);

  panel_vertex_data_ = {0.0f, 0.0f, z_pos_};
  panel_texcoord_data_ = {0.5f, 0.5f};
  panel_color_data_ = {0.0, 0.0, 0.0, panel_alpha_};
  panel_index_data_ = {0};
  for (float i = 0; i < circle_divide; i += 1.0f) {
    float angle = (float) M_PI * 2.0f / circle_divide * i;
    const float x = radius_ * std::cos(angle);
    const float y = radius_ * std::sin(angle);
    panel_vertex_data_.push_back(x);
    panel_vertex_data_.push_back(y);
    panel_vertex_data_.push_back(z_pos_);

    panel_texcoord_data_.push_back(x / radius_ / 2.0f + 0.5f);
    panel_texcoord_data_.push_back(-y / radius_ / 2.0f + 0.5f);

    panel_color_data_.push_back(0.0f);
    panel_color_data_.push_back(0.0f);
    panel_color_data_.push_back(0.0f);
    panel_color_data_.push_back(panel_alpha_);
    panel_index_data_.push_back((GLuint) i + 1);
  }
  panel_index_data_.push_back(1);

  constexpr float arrow_edge = 0.1f;
  const float arrow_length = 0.8f * radius_;

  speed_mags_.resize((size_t) kTraj_, 0.0f);

  constexpr float arrow_ratio = 0.7;
  pointer_vertex_data_ = {0.0f, 0.0f, z_pos_,
                          0.0f, arrow_length, z_pos_,
                          -1 * arrow_edge * arrow_ratio, arrow_length - arrow_edge, z_pos_,
                          arrow_edge * arrow_ratio, arrow_length - arrow_edge, z_pos_};

  for (auto i = 0; i < colors.size(); ++i) {
    for (auto j = 0; j < pointer_vertex_data_.size() / 3; ++j) {
      pointer_color_data_.push_back(colors[i][0]);
      pointer_color_data_.push_back(colors[i][1]);
      pointer_color_data_.push_back(colors[i][2]);
      pointer_color_data_.push_back(1.0f);
    }
  }

  pointer_index_data_ = {0, 1, 1, 2, 1, 3};

  panel_view_matrix_.setToIdentity();
  //panel_view_matrix_.lookAt(QVector3D(0.0f, -radius_/2.0f, z_pos_ + 1.0f), QVector3D(0.0f, radius_/2.0f, z_pos_), QVector3D(0.0f, 1.0f, 0.0f));
  panel_view_matrix_.lookAt(QVector3D(0.0f, 0.0f, z_pos_ + 0.1f),
                            QVector3D(0.0f, 0.0f, z_pos_),
                            QVector3D(0.0f, 1.0f, 0.0f));

  //constexpr float fov = 100.0f;
  panel_projection_matrix_.setToIdentity();
  //panel_projection_matrix_.perspective(fov, 1.0f, 0.0f, 20.0f);
  panel_projection_matrix_.ortho(-radius_, radius_, -radius_, radius_, 0.0f, 10.0f);

  pointer_modelviews_.resize((size_t) kTraj);
  for (auto &mv: pointer_modelviews_) {
    mv = panel_view_matrix_;
  }
}

void OfflineSpeedPanel::InitGL() {
  initializeOpenGLFunctions();

  line_shader_.reset(new QOpenGLShaderProgram());
  CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/line_shader.vert"));
  CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/line_shader.frag"));
  CHECK(line_shader_->link()) << "Speed: can not link line shader";
  CHECK(line_shader_->bind());
  line_shader_->enableAttributeArray("pos");
  line_shader_->enableAttributeArray("v_color");
  line_shader_->release();

  tex_shader_.reset(new QOpenGLShaderProgram());
  CHECK(tex_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/canvas_shader.vert"));
  CHECK(tex_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/canvas_shader.frag"));
  CHECK(tex_shader_->link()) << "Canvas: can not link texture shader";
  CHECK(tex_shader_->bind()) << "Canvas: can not bind texture shader";

  tex_shader_->enableAttributeArray("pos");
  tex_shader_->enableAttributeArray("texcoord");
  tex_shader_->release();

  is_shader_init_ = true;

  glEnable(GL_TEXTURE_2D);
  panel_texture_.reset(new QOpenGLTexture(QImage("../../viewer/resource/images/compass.png")));
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);

  glGenBuffers(1, &panel_vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, panel_vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, panel_vertex_data_.size() * sizeof(GLfloat),
               panel_vertex_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &panel_color_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, panel_color_buffer_);
  glBufferData(GL_ARRAY_BUFFER, panel_color_data_.size() * sizeof(GLfloat),
               panel_color_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &panel_texcoord_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, panel_texcoord_buffer_);
  glBufferData(GL_ARRAY_BUFFER, panel_texcoord_data_.size() * sizeof(GLfloat),
               panel_texcoord_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &panel_index_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, panel_index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, panel_index_data_.size() * sizeof(GLuint),
               panel_index_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &pointer_vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, pointer_vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, pointer_vertex_data_.size() * sizeof(GLfloat),
               pointer_vertex_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &pointer_color_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, pointer_color_buffer_);
  glBufferData(GL_ARRAY_BUFFER, pointer_color_data_.size() * sizeof(GLfloat),
               pointer_color_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &pointer_index_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pointer_index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, pointer_index_data_.size() * sizeof(GLuint),
               pointer_index_data_.data(), GL_STATIC_DRAW);

//		glGenBuffers(1, &device_vertex_buffer_);
//		glBindBuffer(GL_ARRAY_BUFFER, device_vertex_buffer_);
//		glBufferData(GL_ARRAY_BUFFER, device_vertex_data_.size() * sizeof(GLfloat),
//					 device_vertex_data_.data(), GL_STATIC_DRAW);
//
//		glGenBuffers(1, &device_color_buffer_);
//		glBindBuffer(GL_ARRAY_BUFFER, device_color_buffer_);
//		glBufferData(GL_ARRAY_BUFFER, device_color_data_.size() * sizeof(GLfloat),
//					 device_color_data_.data(), GL_STATIC_DRAW);
//
//		glGenBuffers(1, &device_index_buffer_);
//		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, device_index_buffer_);
//		glBufferData(GL_ELEMENT_ARRAY_BUFFER, device_index_data_.size() * sizeof(GLuint),
//					 device_index_data_.data(), GL_STATIC_DRAW);

}

void OfflineSpeedPanel::Render(const Navigation &navigation) {
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_TEXTURE_2D);
  CHECK(tex_shader_->bind());
  tex_shader_->setUniformValue("m_mat", panel_view_matrix_);
  tex_shader_->setUniformValue("p_mat", panel_projection_matrix_);
  glBindBuffer(GL_ARRAY_BUFFER, panel_vertex_buffer_);
  tex_shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3);
  glBindBuffer(GL_ARRAY_BUFFER, panel_texcoord_buffer_);
  tex_shader_->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);

  panel_texture_->bind();
  tex_shader_->setUniformValue("tex_sampler", 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, panel_index_buffer_);
  glDrawElements(GL_TRIANGLE_FAN, (GLsizei) panel_index_data_.size(), GL_UNSIGNED_INT, 0);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);
  panel_texture_->release();
  tex_shader_->release();

  // Render pointers
  CHECK(line_shader_->bind());
  glLineWidth(5.0f);
  for (int i = 0; i < kTraj_; ++i) {
    line_shader_->setUniformValue("p_mat", panel_projection_matrix_);
    line_shader_->setUniformValue("m_mat", pointer_modelviews_[i]);

    glBindBuffer(GL_ARRAY_BUFFER, pointer_vertex_buffer_);
    line_shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3);
    glBindBuffer(GL_ARRAY_BUFFER, pointer_color_buffer_);
    line_shader_->setAttributeBuffer("v_color", GL_FLOAT, i * 16 * sizeof(GLfloat), 4);
    //line_shader_->setAttributeBuffer("v_color", GL_FLOAT, 0, 4);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pointer_index_buffer_);
    glDrawElements(GL_LINES, (GLsizei) pointer_index_data_.size(), GL_UNSIGNED_INT, 0);
  }
  line_shader_->release();

}

/////////////////////////////////////////
// Implementation of legend
LegendRenderer::LegendRenderer(const int width, const int height, const QImage &texture_img)
    : width_((float) width), height_((float) height), z_pos_(-1.0f), texture_img_(texture_img) {

  modelview_.setToIdentity();
  projection_.setToIdentity();
  projection_.ortho(-0.5f * width_, 0.5f * width_, -0.5f * height_, 0.5f * height_,
                    0.0f, 5.0f);

  vertex_data_ = {-0.5f * width_, 0.5f * height_, z_pos_,
                  0.5f * width_, 0.5f * height_, z_pos_,
                  0.5f * width_, -0.5f * height_, z_pos_,
                  -0.5f * width_, -0.5f * height_, z_pos_};
  texcoord_data_ = {0.0f, 0.0f,
                    1.0f, 0.0f,
                    1.0f, 1.0f,
                    0.0f, 1.0f};
  index_data_ = {0, 1, 2, 2, 3, 0};
}
void LegendRenderer::InitGL() {
  initializeOpenGLFunctions();

  tex_shader_.reset(new QOpenGLShaderProgram());
  CHECK(tex_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/canvas_shader.vert"));
  CHECK(tex_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/canvas_shader.frag"));
  CHECK(tex_shader_->link()) << "Canvas: can not link texture shader";
  CHECK(tex_shader_->bind()) << "Canvas: can not bind texture shader";

  tex_shader_->enableAttributeArray("pos");
  tex_shader_->enableAttributeArray("texcoord");
  tex_shader_->release();

  glEnable(GL_TEXTURE_2D);
  texture_.reset(new QOpenGLTexture(QImage(texture_img_)));
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);

  glGenBuffers(1, &vertex_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  glBufferData(GL_ARRAY_BUFFER, vertex_data_.size() * sizeof(GLfloat),
               vertex_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &texcoord_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
  glBufferData(GL_ARRAY_BUFFER, texcoord_data_.size() * sizeof(GLfloat),
               texcoord_data_.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &index_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data_.size() * sizeof(GLuint),
               index_data_.data(), GL_STATIC_DRAW);
}
void LegendRenderer::Render(const Navigation &navigation) {
  CHECK(tex_shader_->bind());
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  tex_shader_->setUniformValue("p_mat", projection_);
  tex_shader_->setUniformValue("m_mat", modelview_);
  texture_->bind();
  tex_shader_->setUniformValue("tex_sampler", 0);

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
  tex_shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3);
  glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
  tex_shader_->setAttributeBuffer("texcoord", GL_FLOAT, 0, 2);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glDrawElements(GL_TRIANGLES, (GLsizei) index_data_.size(), GL_UNSIGNED_INT, 0);

  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);
  texture_->release();
  tex_shader_->release();
}

}//namespace IMUProject
