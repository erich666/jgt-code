/*
 * Ares Lagae and Philip Dutré.
 * An efficient ray-quadrilateral intersection test.
 * Journal of graphics tools, 10(4):23-32, 2005
 *
 * This program is a minimal ray tracer. It ray traces a single quad, covered
 * with a checkerboard texture.
 *
 * The classes provided in this file are not complete, only operations needed
 * to implement the intersection algorithm and ray tracer are provided.
 *
 * Copyright Ares Lagae (ares lagae at cs kuleuven ac be), 2004.
 *
 * Permission is hereby granted to use, copy, modify, and distribute this
 * software (or portions thereof) for any purpose, without fee.
 *
 * Ares Lagae makes no representations about the suitability of this
 * software for any purpose. It is provided "as is" without express or
 * implied warranty.
 *
 * See http://www.acm.org/jgt/papers/LagaeDutre05
 * for the most recent version of this file and additional documentation.
 * 
 * Revision history
 *  2004-10-08  initial version
 *  2004-03-11  minor changes
 *  2006-04-14  minor changes
 */ 

#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

typedef double real;

class vector
{
public:
  vector(real x, real y, real z) { xyz[0] = x; xyz[1] = y; xyz[2] = z; }
  real x() const { return xyz[0]; }
  real y() const { return xyz[1]; }
  real z() const { return xyz[2]; }
private:
  real xyz[3];
};

inline real dot(const vector& lhs, const vector& rhs)
{
  return (lhs.x() * rhs.x()) +  (lhs.y() * rhs.y()) +  (lhs.z() * rhs.z());
}

inline vector cross(const vector& lhs, const vector& rhs)
{
  return vector((lhs.y() * rhs.z()) - (lhs.z() * rhs.y()),
                (lhs.z() * rhs.x()) - (lhs.x() * rhs.z()),
                (lhs.x() * rhs.y()) - (lhs.y() * rhs.x()));
}

class point
{
public:
  point() {}
  point(real x, real y, real z) { xyz[0] = x; xyz[1] = y; xyz[2] = z; }
  real x() const { return xyz[0]; }  
  real y() const { return xyz[1]; }
  real z() const { return xyz[2]; }
private:
  real xyz[3];
};

inline vector operator-(const point& lhs, const point& rhs)
{
  return vector(lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z());
}

class ray
{
public:
  ray(const point& origin, const vector& direction)
    : origin_(origin), direction_(direction) {}
  const point& origin() const { return origin_; }
  const vector& direction() const { return direction_; }
private:
  point origin_;
  vector direction_;
};

class quadrilateral
{
public:
  quadrilateral(const point& v_00, const point& v_10,
    const point& v_11, const point& v_01)
  {
    vertices[0] = v_00;
    vertices[1] = v_10;
    vertices[2] = v_11;
    vertices[3] = v_01;
  }
  const point& v_00() const { return vertices[0]; }
  const point& v_10() const { return vertices[1]; }
  const point& v_11() const { return vertices[2]; }
  const point& v_01() const { return vertices[3]; }
private:
  point vertices[4];
};

class rgb_color
{
public:
  rgb_color() {}
  rgb_color(real r, real g, real b) { rgb[0] = r; rgb[1] = g; rgb[2] = b; }
  real r() const { return rgb[0]; }  
  real g() const { return rgb[1]; }
  real b() const { return rgb[2]; }
private:
  real rgb[3];
};

class checkerboard_texture
{
public:
  checkerboard_texture(unsigned num_rows, unsigned num_cols,
      const rgb_color& black_color, const rgb_color& white_color)
    : num_rows_(num_rows), num_cols_(num_cols), black_color_(black_color),
      white_color_(white_color) {}
  const rgb_color& operator()(real u, real v) const
  {
    return (unsigned(u*num_cols_) + unsigned(v*num_rows_)) % 2 ?
      white_color_ : black_color_;
  }
private:
  unsigned num_rows_, num_cols_;
  rgb_color black_color_, white_color_;
};

class image
{
public:
  image(std::size_t width, std::size_t height, const rgb_color& color)
    : width_(width), height_(height)
  {
    pixels_ = new rgb_color[width_ * height_];
    std::fill_n(pixels_, width_ * height_, color);
  }
  ~image() { delete [] pixels_; }
  std::size_t width() const { return width_; }
  std::size_t height() const { return height_; }
  const rgb_color& operator()(std::size_t row, std::size_t col) const
  {
    return pixels_[(row * width_) + col];
  }
  rgb_color& operator()(std::size_t row, std::size_t col)
  {
    return pixels_[(row * width_) + col];
  }
  bool write_ppm(std::ostream& os) const;
private:
  image(const image&);
  image& operator=(const image&);
  std::size_t width_, height_;
  rgb_color* pixels_;
};

bool image::write_ppm(std::ostream& os) const
{
  os << "P3\n" << width() << ' ' << height() << '\n' << 255 << '\n';

  for (std::size_t row = 0; row < height(); ++row) {

    for (std::size_t col = 0; col < width(); ++col) {

      const rgb_color& pixel = (*this)(row, col);
      os << static_cast<int>(pixel.r() * 255) << ' '
         << static_cast<int>(pixel.g() * 255) << ' '
         << static_cast<int>(pixel.b() * 255) << ' ';
    }
    os << '\n';
  }
  return os;
}

bool intersect_quadrilateral_ray(const quadrilateral& q,
  const ray& r, real& u, real& v, real& t)
{
  static const real eps = real(10e-6);

  // Rejects rays that are parallel to Q, and rays that intersect the plane of
  // Q either on the left of the line V00V01 or on the right of the line V00V10.

  vector E_01 = q.v_10() - q.v_00();
  vector E_03 = q.v_01() - q.v_00();
  vector P = cross(r.direction(), E_03);
  real det = dot(E_01, P);
  if (std::abs(det) < eps) return false;
  real inv_det = real(1.0) / det;
  vector T = r.origin() - q.v_00();
  real alpha = dot(T, P) * inv_det;
  if (alpha < real(0.0)) return false;
  // if (alpha > real(1.0)) return false; // Uncomment if VR is used.
  vector Q = cross(T, E_01);
  real beta = dot(r.direction(), Q) * inv_det;
  if (beta < real(0.0)) return false; 
  // if (beta > real(1.0)) return false; // Uncomment if VR is used.

  if ((alpha + beta) > real(1.0)) {

    // Rejects rays that intersect the plane of Q either on the
    // left of the line V11V10 or on the right of the line V11V01.

    vector E_23 = q.v_01() - q.v_11();
    vector E_21 = q.v_10() - q.v_11();
    vector P_prime = cross(r.direction(), E_21);
    real det_prime = dot(E_23, P_prime);
    if (std::abs(det_prime) < eps) return false;
    real inv_det_prime = real(1.0) / det_prime;
    vector T_prime = r.origin() - q.v_11();
    real alpha_prime = dot(T_prime, P_prime) * inv_det_prime;
    if (alpha_prime < real(0.0)) return false;
    vector Q_prime = cross(T_prime, E_23);
    real beta_prime = dot(r.direction(), Q_prime) * inv_det_prime;
    if (beta_prime < real(0.0)) return false;
  }

  // Compute the ray parameter of the intersection point, and
  // reject the ray if it does not hit Q.

  t = dot(E_03, Q) * inv_det;
  if (t < real(0.0)) return false; 

  // Compute the barycentric coordinates of the fourth vertex.
  // These do not depend on the ray, and can be precomputed
  // and stored with the quadrilateral.  

  real alpha_11, beta_11;
  vector E_02 = q.v_11() - q.v_00();
  vector n = cross(E_01, E_03);

  if ((std::abs(n.x()) >= std::abs(n.y()))
    && (std::abs(n.x()) >= std::abs(n.z()))) {

    alpha_11 = ((E_02.y() * E_03.z()) - (E_02.z() * E_03.y())) / n.x();
    beta_11  = ((E_01.y() * E_02.z()) - (E_01.z() * E_02.y())) / n.x();
  }
  else if ((std::abs(n.y()) >= std::abs(n.x()))
    && (std::abs(n.y()) >= std::abs(n.z()))) {  

    alpha_11 = ((E_02.z() * E_03.x()) - (E_02.x() * E_03.z())) / n.y();
    beta_11  = ((E_01.z() * E_02.x()) - (E_01.x() * E_02.z())) / n.y();
  }
  else {

    alpha_11 = ((E_02.x() * E_03.y()) - (E_02.y() * E_03.x())) / n.z();
    beta_11  = ((E_01.x() * E_02.y()) - (E_01.y() * E_02.x())) / n.z();
  }

  // Compute the bilinear coordinates of the intersection point.

  if (std::abs(alpha_11 - real(1.0)) < eps) {    

    // Q is a trapezium.
    u = alpha;
    if (std::abs(beta_11 - real(1.0)) < eps) v = beta; // Q is a parallelogram.
    else v = beta / ((u * (beta_11 - real(1.0))) + real(1.0)); // Q is a trapezium.
  }
  else if (std::abs(beta_11 - real(1.0)) < eps) {

    // Q is a trapezium.
    v = beta;
    u = alpha / ((v * (alpha_11 - real(1.0))) + real(1.0));
  }
  else {

    real A = real(1.0) - beta_11;
    real B = (alpha * (beta_11 - real(1.0)))
      - (beta * (alpha_11 - real(1.0))) - real(1.0);
    real C = alpha;
    real D = (B * B) - (real(4.0) * A * C);
    real Q = real(-0.5) * (B + ((B < real(0.0) ? real(-1.0) : real(1.0))
      * std::sqrt(D)));
    u = Q / A;
    if ((u < real(0.0)) || (u > real(1.0))) u = C / Q;
    v = beta / ((u * (beta_11 - real(1.0))) + real(1.0)); 
  }

  return true;
}

int main()
{
  quadrilateral q(point( 0.49421906944, 0.081285633543, 0.100104041766),
                  point( 1.00316508089, 0.530985148652, 0.629377264874),
                  point( 0.50578093056, 0.918714366457, 0.899895958234),
                  point(-0.01235416806, 0.590487788947, 0.484479525901)); 

  image img(256, 256, rgb_color(0, 0, 0));

  checkerboard_texture texture(4, 4, rgb_color(1, 0, 0), rgb_color(1, 1, 1));

  for (std::size_t row = 0; row < img.height(); ++row) {

    real y = (real(img.height() - row - 1) + real(0.5)) / img.height();

    for (std::size_t col = 0; col < img.width(); ++col) {

      real x = (real(col) + real(0.5)) / img.width();
      ray r(point(x, y, 10), vector(0, 0, -1));

      real u, v, t;
      if (intersect_quadrilateral_ray(q, r, u, v, t)) {
        img(row, col) = texture(u, v);
      }
    }
  }

  std::ofstream os("image.ppm");
  img.write_ppm(os);

  return 0;
}
