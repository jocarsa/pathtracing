#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <opencv2/opencv.hpp>

// Math utility functions and types
struct Vec3 {
    double x, y, z;

    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }
    
    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const { 
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); 
    }
    
    double length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalize() const { 
        double len = length();
        if (len > 0) return *this / len;
        return *this;
    }
};

// Ray definition
struct Ray {
    Vec3 origin;
    Vec3 direction;
    
    // Add default constructor
    Ray() : origin(Vec3(0, 0, 0)), direction(Vec3(0, 0, 1)) {}
    
    Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d.normalize()) {}
    
    Vec3 point_at(double t) const { return origin + direction * t; }
};

// Materials
enum class MaterialType { DIFFUSE, METAL, DIELECTRIC, EMISSIVE };

struct Material {
    MaterialType type;
    Vec3 albedo;
    Vec3 emission;
    double roughness;
    double refraction_index;
    
    Material(MaterialType t = MaterialType::DIFFUSE, 
             const Vec3& a = Vec3(0.8, 0.8, 0.8),
             const Vec3& e = Vec3(0, 0, 0),
             double r = 0.0,
             double ri = 1.0)
        : type(t), albedo(a), emission(e), roughness(r), refraction_index(ri) {}
};

// Hit record for intersections
struct HitRecord {
    double t;
    Vec3 point;
    Vec3 normal;
    Material material;
    
    HitRecord() : t(std::numeric_limits<double>::max()) {}
};

// Abstract base class for all hittable objects
class Hittable {
public:
    virtual bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const = 0;
    virtual ~Hittable() = default;
};

// Sphere implementation
class Sphere : public Hittable {
public:
    Vec3 center;
    double radius;
    Material material;
    
    Sphere(const Vec3& c, double r, const Material& m)
        : center(c), radius(r), material(m) {}
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const override {
        Vec3 oc = ray.origin - center;
        double a = ray.direction.dot(ray.direction);
        double b = oc.dot(ray.direction);
        double c = oc.dot(oc) - radius * radius;
        double discriminant = b * b - a * c;
        
        if (discriminant > 0) {
            double temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.point = ray.point_at(rec.t);
                rec.normal = (rec.point - center) / radius;
                rec.material = material;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.point = ray.point_at(rec.t);
                rec.normal = (rec.point - center) / radius;
                rec.material = material;
                return true;
            }
        }
        return false;
    }
};

// Plane implementation for the ground
class Plane : public Hittable {
public:
    Vec3 point;
    Vec3 normal;
    Material material;
    
    Plane(const Vec3& p, const Vec3& n, const Material& m)
        : point(p), normal(n.normalize()), material(m) {}
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const override {
        double denom = normal.dot(ray.direction);
        if (std::abs(denom) > 1e-6) {
            Vec3 p0l0 = point - ray.origin;
            double t = p0l0.dot(normal) / denom;
            if (t >= t_min && t <= t_max) {
                rec.t = t;
                rec.point = ray.point_at(t);
                rec.normal = normal;
                rec.material = material;
                return true;
            }
        }
        return false;
    }
};

// Triangle implementation for the cube
class Triangle : public Hittable {
public:
    Vec3 v0, v1, v2;
    Material material;
    
    Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2, const Material& m)
        : v0(v0), v1(v1), v2(v2), material(m) {}
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const override {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = ray.direction.cross(edge2);
        double a = edge1.dot(h);
        
        if (std::abs(a) < 1e-6) {
            return false;  // Ray is parallel to triangle
        }
        
        double f = 1.0 / a;
        Vec3 s = ray.origin - v0;
        double u = f * s.dot(h);
        
        if (u < 0.0 || u > 1.0) {
            return false;
        }
        
        Vec3 q = s.cross(edge1);
        double v = f * ray.direction.dot(q);
        
        if (v < 0.0 || u + v > 1.0) {
            return false;
        }
        
        double t = f * edge2.dot(q);
        
        if (t > t_min && t < t_max) {
            rec.t = t;
            rec.point = ray.point_at(t);
            rec.normal = edge1.cross(edge2).normalize();
            rec.material = material;
            return true;
        } else {
            return false;
        }
    }
};

// Cube class composed of triangles
class Cube : public Hittable {
private:
    std::vector<Triangle> triangles;
    Vec3 center;
    double size;
    
public:
    Cube(const Vec3& center, double size, const Material& material) 
        : center(center), size(size) {
        // Define the 8 corners of the cube
        double half = size / 2.0;
        Vec3 p1 = Vec3(center.x - half, center.y - half, center.z - half);
        Vec3 p2 = Vec3(center.x + half, center.y - half, center.z - half);
        Vec3 p3 = Vec3(center.x + half, center.y + half, center.z - half);
        Vec3 p4 = Vec3(center.x - half, center.y + half, center.z - half);
        Vec3 p5 = Vec3(center.x - half, center.y - half, center.z + half);
        Vec3 p6 = Vec3(center.x + half, center.y - half, center.z + half);
        Vec3 p7 = Vec3(center.x + half, center.y + half, center.z + half);
        Vec3 p8 = Vec3(center.x - half, center.y + half, center.z + half);
        
        // Create 12 triangles (2 for each face)
        // Front face
        triangles.push_back(Triangle(p1, p2, p3, material));
        triangles.push_back(Triangle(p1, p3, p4, material));
        
        // Back face
        triangles.push_back(Triangle(p5, p7, p6, material));
        triangles.push_back(Triangle(p5, p8, p7, material));
        
        // Left face
        triangles.push_back(Triangle(p1, p4, p8, material));
        triangles.push_back(Triangle(p1, p8, p5, material));
        
        // Right face
        triangles.push_back(Triangle(p2, p6, p7, material));
        triangles.push_back(Triangle(p2, p7, p3, material));
        
        // Top face
        triangles.push_back(Triangle(p4, p3, p7, material));
        triangles.push_back(Triangle(p4, p7, p8, material));
        
        // Bottom face
        triangles.push_back(Triangle(p1, p5, p6, material));
        triangles.push_back(Triangle(p1, p6, p2, material));
    }
    
    void rotate_y(double angle) {
        // Create new triangles with rotated vertices
        std::vector<Triangle> rotated_triangles;
        double cos_angle = std::cos(angle);
        double sin_angle = std::sin(angle);
        
        for (const auto& triangle : triangles) {
            // Rotate each vertex of the triangle around the y-axis
            Vec3 v0 = triangle.v0 - center;
            Vec3 v1 = triangle.v1 - center;
            Vec3 v2 = triangle.v2 - center;
            
            Vec3 v0_rotated = Vec3(
                v0.x * cos_angle + v0.z * sin_angle,
                v0.y,
                -v0.x * sin_angle + v0.z * cos_angle
            ) + center;
            
            Vec3 v1_rotated = Vec3(
                v1.x * cos_angle + v1.z * sin_angle,
                v1.y,
                -v1.x * sin_angle + v1.z * cos_angle
            ) + center;
            
            Vec3 v2_rotated = Vec3(
                v2.x * cos_angle + v2.z * sin_angle,
                v2.y,
                -v2.x * sin_angle + v2.z * cos_angle
            ) + center;
            
            rotated_triangles.push_back(Triangle(v0_rotated, v1_rotated, v2_rotated, triangle.material));
        }
        
        triangles = rotated_triangles;
    }
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const override {
        HitRecord temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;
        
        for (const auto& triangle : triangles) {
            if (triangle.hit(ray, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        
        return hit_anything;
    }
};

// Scene definition
class Scene {
public:
    std::vector<std::shared_ptr<Hittable>> objects;
    
    void add(std::shared_ptr<Hittable> object) {
        objects.push_back(object);
    }
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;
        
        for (const auto& object : objects) {
            if (object->hit(ray, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        
        return hit_anything;
    }
};

// Camera implementation
class Camera {
public:
    Vec3 position;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    
    Camera(const Vec3& position, const Vec3& look_at, const Vec3& up, 
           double fov, double aspect_ratio) 
        : position(position) {
        
        double theta = fov * M_PI / 180.0;
        double half_height = tan(theta / 2.0);
        double half_width = aspect_ratio * half_height;
        
        Vec3 w = (position - look_at).normalize();
        Vec3 u = up.cross(w).normalize();
        Vec3 v = w.cross(u);
        
        lower_left_corner = position - u * half_width - v * half_height - w;
        horizontal = u * 2.0 * half_width;
        vertical = v * 2.0 * half_height;
    }
    
    Ray get_ray(double s, double t) const {
        return Ray(position, lower_left_corner + horizontal * s + vertical * t - position);
    }
};

// Random number generation
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(0.0, 1.0);

double random_double() {
    return dis(gen);
}

Vec3 random_in_unit_sphere() {
    while (true) {
        Vec3 p(random_double() * 2.0 - 1.0, 
               random_double() * 2.0 - 1.0, 
               random_double() * 2.0 - 1.0);
        if (p.dot(p) < 1.0) {
            return p;
        }
    }
}

Vec3 random_in_hemisphere(const Vec3& normal) {
    Vec3 in_unit_sphere = random_in_unit_sphere();
    if (in_unit_sphere.dot(normal) > 0.0) {
        return in_unit_sphere;
    } else {
        return in_unit_sphere * -1.0;
    }
}

// Custom clamp function since std::clamp requires C++17
double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Sky (environment) color
Vec3 sky_color(const Ray& ray) {
    Vec3 unit_direction = ray.direction.normalize();
    double t = 0.5 * (unit_direction.y + 1.0);
    return Vec3(1.0, 1.0, 1.0) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t;
}

// Path tracer function
Vec3 trace(const Ray& ray, const Scene& scene, int depth) {
    if (depth <= 0) {
        return Vec3(0, 0, 0);
    }
    
    HitRecord rec;
    if (scene.hit(ray, 0.001, std::numeric_limits<double>::infinity(), rec)) {
        Ray scattered;
        Vec3 attenuation;
        Vec3 emitted = rec.material.emission;
        
        // Handle different material types
        switch (rec.material.type) {
            case MaterialType::DIFFUSE: {
                // Lambertian diffuse reflection
                Vec3 target = rec.point + rec.normal + random_in_hemisphere(rec.normal);
                scattered = Ray(rec.point, target - rec.point);
                attenuation = rec.material.albedo;
                return emitted + attenuation * trace(scattered, scene, depth - 1);
            }
            case MaterialType::METAL: {
                // Metal reflection
                Vec3 reflected = ray.direction - rec.normal * 2 * ray.direction.dot(rec.normal);
                reflected = reflected + random_in_unit_sphere() * rec.material.roughness;
                scattered = Ray(rec.point, reflected);
                attenuation = rec.material.albedo;
                
                if (scattered.direction.dot(rec.normal) > 0) {
                    return emitted + attenuation * trace(scattered, scene, depth - 1);
                } else {
                    return emitted;
                }
            }
            case MaterialType::EMISSIVE: {
                // Simply return emission
                return emitted;
            }
            default:
                return Vec3(0, 0, 0);
        }
    }
    
    // If no hit, return sky color
    return sky_color(ray);
}

// Main function
int main() {
    // Image settings
    const int width = 800;
    const int height = 600;
    const int samples_per_pixel = 50;
    const int max_depth = 10;
    const double aspect_ratio = static_cast<double>(width) / height;
    
    // Camera setup
    Camera camera(Vec3(0, 2, 6), Vec3(0, 1, 0), Vec3(0, 1, 0), 60.0, aspect_ratio);
    
    // Scene creation
    Scene scene;
    
    // Ground plane
    scene.add(std::make_shared<Plane>(
        Vec3(0, 0, 0), Vec3(0, 1, 0), 
        Material(MaterialType::DIFFUSE, Vec3(0.8, 0.8, 0.8))
    ));
    
    // Rotating cube
    auto cube_material = Material(MaterialType::METAL, Vec3(0.7, 0.3, 0.3), Vec3(0, 0, 0), 0.1);
    auto cube = std::make_shared<Cube>(Vec3(0, 1.5, 0), 2.0, cube_material);
    
    // Sun (directional light, simulated as a distant sphere)
    scene.add(std::make_shared<Sphere>(
        Vec3(50, 100, -30), 20.0, 
        Material(MaterialType::EMISSIVE, Vec3(1.0, 1.0, 0.8), Vec3(15.0, 15.0, 12.0))
    ));
    
    // OpenCV image buffer
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Animation loop
    const int num_frames = 60;
    for (int frame = 0; frame < num_frames; ++frame) {
        // Create a new cube with rotation for this frame
        auto rotated_cube = std::make_shared<Cube>(Vec3(0, 1.5, 0), 2.0, cube_material);
        double angle = 2.0 * M_PI * frame / num_frames;
        rotated_cube->rotate_y(angle);
        
        // Update scene with new cube
        scene.objects.clear();
        
        // Add ground
        scene.add(std::make_shared<Plane>(
            Vec3(0, 0, 0), Vec3(0, 1, 0), 
            Material(MaterialType::DIFFUSE, Vec3(0.8, 0.8, 0.8))
        ));
        
        // Add cube
        scene.add(rotated_cube);
        
        // Add sun
        scene.add(std::make_shared<Sphere>(
            Vec3(50, 100, -30), 20.0, 
            Material(MaterialType::EMISSIVE, Vec3(1.0, 1.0, 0.8), Vec3(15.0, 15.0, 12.0))
        ));
        
        // Render scene
        std::cout << "Rendering frame " << frame + 1 << "/" << num_frames << "..." << std::endl;
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                Vec3 color(0, 0, 0);
                
                // Super-sampling for anti-aliasing
                for (int s = 0; s < samples_per_pixel; ++s) {
                    double u = (i + random_double()) / width;
                    double v = (j + random_double()) / height;
                    Ray ray = camera.get_ray(u, 1.0 - v);  // Invert v for OpenCV
                    color = color + trace(ray, scene, max_depth);
                }
                
                // Average the samples and apply gamma correction
                color = color / samples_per_pixel;
                color = Vec3(sqrt(color.x), sqrt(color.y), sqrt(color.z));
                
                // Write to image buffer (using our own clamp function)
                image.at<cv::Vec3b>(j, i) = cv::Vec3b(
                    static_cast<unsigned char>(255.99 * clamp(color.z, 0.0, 1.0)),
                    static_cast<unsigned char>(255.99 * clamp(color.y, 0.0, 1.0)),
                    static_cast<unsigned char>(255.99 * clamp(color.x, 0.0, 1.0))
                );
            }
        }
        
        // Display the rendered image
        cv::imshow("Path Tracer", image);
        
        // Save the frame
        std::string filename = "frame_" + std::to_string(frame) + ".png";
        cv::imwrite(filename, image);
        
        // Wait for key press (30ms) - adjust for animation speed
        cv::waitKey(30);
    }
    
    // Wait for a key press before closing
    std::cout << "Rendering complete. Press any key to exit." << std::endl;
    cv::waitKey(0);
    
    return 0;
}
