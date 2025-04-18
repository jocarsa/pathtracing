#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

// ----------------------------------------------------------------
// CUDA Error Checking Macro
// ----------------------------------------------------------------
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Math utility functions and types
struct Vec3 {
    double x, y, z;

    __host__ __device__ Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }
    
    __host__ __device__ double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3 cross(const Vec3& v) const { 
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); 
    }
    
    __host__ __device__ double length() const { return sqrt(x * x + y * y + z * z); }
    __host__ __device__ Vec3 normalize() const { 
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
    __host__ __device__ Ray() : origin(Vec3(0, 0, 0)), direction(Vec3(0, 0, 1)) {}
    
    __host__ __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d.normalize()) {}
    
    __host__ __device__ Vec3 point_at(double t) const { return origin + direction * t; }
};

// Materials
enum class MaterialType { DIFFUSE, METAL, DIELECTRIC, EMISSIVE };

struct Material {
    MaterialType type;
    Vec3 albedo;
    Vec3 emission;
    double roughness;
    double refraction_index;
    
    __host__ __device__ Material(MaterialType t = MaterialType::DIFFUSE, 
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
    
    __host__ __device__ HitRecord() : t(INFINITY) {}
};

// Forward declarations of device helper functions
__device__ Vec3 random_in_unit_sphere(curandState* rand_state);
__device__ Vec3 random_in_hemisphere(const Vec3& normal, curandState* rand_state);
__device__ Vec3 sky_color(const Ray& ray);
__device__ double clamp(double x, double min, double max);

// Device code for ray-sphere intersection
__device__ bool hit_sphere(const Ray& ray, const Vec3& center, double radius, 
                          const Material& material, double t_min, double t_max, HitRecord& rec) {
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

// Device code for ray-plane intersection
__device__ bool hit_plane(const Ray& ray, const Vec3& point, const Vec3& normal, 
                         const Material& material, double t_min, double t_max, HitRecord& rec) {
    double denom = normal.dot(ray.direction);
    if (fabs(denom) > 1e-6) {
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

// Device code for ray-triangle intersection
__device__ bool hit_triangle(const Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2,
                            const Material& material, double t_min, double t_max, HitRecord& rec) {
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = ray.direction.cross(edge2);
    double a = edge1.dot(h);
    
    if (fabs(a) < 1e-6) {
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
    }
    
    return false;
}

// Scene structure for GPU
struct GPUScene {
    // Ground plane
    Vec3 plane_point;
    Vec3 plane_normal;
    Material plane_material;
    
    // Sun (emissive sphere)
    Vec3 sun_center;
    double sun_radius;
    Material sun_material;
    
    // Cube data (12 triangles)
    Vec3 cube_vertices[8];
    Material cube_material;
    
    __host__ __device__ GPUScene() {}
};

// Device function to check if a ray hits anything in the scene
__device__ bool scene_hit(const Ray& ray, const GPUScene& scene, double t_min, double t_max, HitRecord& rec) {
    HitRecord temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    
    // Check ground plane
    if (hit_plane(ray, scene.plane_point, scene.plane_normal, scene.plane_material, t_min, closest_so_far, temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
    }
    
    // Check sun sphere
    if (hit_sphere(ray, scene.sun_center, scene.sun_radius, scene.sun_material, t_min, closest_so_far, temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
    }
    
    // Check cube (12 triangles)
    // Define the indices for the 12 triangles (2 per face)
    int triangle_indices[12][3] = {
        {0, 1, 2}, {0, 2, 3},  // Front face
        {4, 6, 5}, {4, 7, 6},  // Back face
        {0, 3, 7}, {0, 7, 4},  // Left face
        {1, 5, 6}, {1, 6, 2},  // Right face
        {3, 2, 6}, {3, 6, 7},  // Top face
        {0, 4, 5}, {0, 5, 1}   // Bottom face
    };
    
    for (int i = 0; i < 12; i++) {
        if (hit_triangle(ray, 
                         scene.cube_vertices[triangle_indices[i][0]], 
                         scene.cube_vertices[triangle_indices[i][1]], 
                         scene.cube_vertices[triangle_indices[i][2]],
                         scene.cube_material, 
                         t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    return hit_anything;
}

// Device function to generate random number
__device__ double random_double(curandState* rand_state) {
    return curand_uniform_double(rand_state);
}

// Device function to generate random point in unit sphere
__device__ Vec3 random_in_unit_sphere(curandState* rand_state) {
    while (true) {
        Vec3 p(random_double(rand_state) * 2.0 - 1.0, 
               random_double(rand_state) * 2.0 - 1.0, 
               random_double(rand_state) * 2.0 - 1.0);
        if (p.dot(p) < 1.0) {
            return p;
        }
    }
}

// Device function to generate random point in hemisphere
__device__ Vec3 random_in_hemisphere(const Vec3& normal, curandState* rand_state) {
    Vec3 in_unit_sphere = random_in_unit_sphere(rand_state);
    if (in_unit_sphere.dot(normal) > 0.0) {
        return in_unit_sphere;
    } else {
        return in_unit_sphere * -1.0;
    }
}

// Device function for sky color
__device__ Vec3 sky_color(const Ray& ray) {
    Vec3 unit_direction = ray.direction.normalize();
    double t = 0.5 * (unit_direction.y + 1.0);
    return Vec3(1.0, 1.0, 1.0) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t;
}

// Device function to clamp values
__device__ double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Path tracer kernel function
__device__ Vec3 trace(const Ray& ray, const GPUScene& scene, int depth, curandState* rand_state) {
    Ray current_ray = ray;
    Vec3 current_attenuation(1.0, 1.0, 1.0);
    Vec3 final_color(0.0, 0.0, 0.0);
    
    for (int i = 0; i < depth; i++) {
        HitRecord rec;
        if (scene_hit(current_ray, scene, 0.001, INFINITY, rec)) {
            Vec3 emitted = rec.material.emission;
            final_color = final_color + current_attenuation * emitted;
            
            // Handle different material types
            if (rec.material.type == MaterialType::DIFFUSE) {
                // Lambertian diffuse reflection
                Vec3 target = rec.point + rec.normal + random_in_hemisphere(rec.normal, rand_state);
                Ray scattered(rec.point, target - rec.point);
                Vec3 attenuation = rec.material.albedo;
                
                current_attenuation = current_attenuation * attenuation;
                current_ray = scattered;
            } 
            else if (rec.material.type == MaterialType::METAL) {
                // Metal reflection
                Vec3 reflected = current_ray.direction - rec.normal * 2 * current_ray.direction.dot(rec.normal);
                reflected = reflected + random_in_unit_sphere(rand_state) * rec.material.roughness;
                Ray scattered(rec.point, reflected);
                Vec3 attenuation = rec.material.albedo;
                
                if (scattered.direction.dot(rec.normal) > 0) {
                    current_attenuation = current_attenuation * attenuation;
                    current_ray = scattered;
                } else {
                    break;
                }
            }
            else if (rec.material.type == MaterialType::EMISSIVE) {
                // Simply return emission
                break;
            }
            else {
                break;
            }
        } else {
            final_color = final_color + current_attenuation * sky_color(current_ray);
            break;
        }
        
        // Russian roulette for path termination
        if (i > 3) {
            double p = max(current_attenuation.x, max(current_attenuation.y, current_attenuation.z));
            if (random_double(rand_state) > p) {
                break;
            }
            current_attenuation = current_attenuation / p;
        }
    }
    
    return final_color;
}

// Kernel to initialize CUDA random states
__global__ void init_rand(curandState *rand_state, int width, int height, unsigned long seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    
    int pixel_index = j * width + i;
    // Each thread gets same seed, different sequence number
    curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
}

// Kernel to render scene
__global__ void render_kernel(Vec3 *output, int width, int height, int samples_per_pixel, int max_depth,
                             curandState *rand_state, GPUScene scene, 
                             Vec3 camera_pos, Vec3 lower_left_corner, Vec3 horizontal, Vec3 vertical) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    
    int pixel_index = j * width + i;
    curandState local_rand_state = rand_state[pixel_index];
    
    Vec3 color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s++) {
        double u = (i + random_double(&local_rand_state)) / width;
        double v = (j + random_double(&local_rand_state)) / height;
        Ray ray(camera_pos, lower_left_corner + horizontal * u + vertical * (1.0 - v) - camera_pos);
        color = color + trace(ray, scene, max_depth, &local_rand_state);
    }
    
    // Average the samples and apply gamma correction
    color = color / samples_per_pixel;
    color = Vec3(sqrt(color.x), sqrt(color.y), sqrt(color.z));
    
    // Clamp values
    color.x = clamp(color.x, 0.0, 1.0);
    color.y = clamp(color.y, 0.0, 1.0);
    color.z = clamp(color.z, 0.0, 1.0);
    
    output[pixel_index] = color;
    rand_state[pixel_index] = local_rand_state;
}

int main() {
    // Image settings
    const int width = 800;
    const int height = 600;
    const int samples_per_pixel = 50;
    const int max_depth = 10;
    const double aspect_ratio = static_cast<double>(width) / height;
    
    // Camera parameters
    Vec3 camera_pos(0, 2, 6);
    Vec3 look_at(0, 1, 0);
    Vec3 up(0, 1, 0);
    double fov = 60.0;
    
    // Calculate camera parameters
    double theta = fov * M_PI / 180.0;
    double half_height = tan(theta / 2.0);
    double half_width = aspect_ratio * half_height;
    
    Vec3 w = (camera_pos - look_at).normalize();
    Vec3 u = up.cross(w).normalize();
    Vec3 v = w.cross(u);
    
    Vec3 lower_left_corner = camera_pos - u * half_width - v * half_height - w;
    Vec3 horizontal = u * 2.0 * half_width;
    Vec3 vertical = v * 2.0 * half_height;
    
    // Setup CUDA device and allocate unified memory
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Running on GPU: " << prop.name << std::endl;
    
    // Allocate device memory for output image
    Vec3 *output_image;
    CUDA_CHECK(cudaMallocManaged(&output_image, width * height * sizeof(Vec3)));
    
    // Initialize curand states
    curandState *d_rand_state;
    CUDA_CHECK(cudaMalloc(&d_rand_state, width * height * sizeof(curandState)));
    
    // Setup scene on CPU
    GPUScene cpu_scene;
    
    // Ground plane
    cpu_scene.plane_point = Vec3(0, 0, 0);
    cpu_scene.plane_normal = Vec3(0, 1, 0);
    cpu_scene.plane_material = Material(MaterialType::DIFFUSE, Vec3(0.8, 0.8, 0.8));
    
    // Sun (emissive sphere)
    cpu_scene.sun_center = Vec3(50, 100, -30);
    cpu_scene.sun_radius = 20.0;
    cpu_scene.sun_material = Material(MaterialType::EMISSIVE, Vec3(1.0, 1.0, 0.8), Vec3(15.0, 15.0, 12.0));
    
    // Cube material
    cpu_scene.cube_material = Material(MaterialType::METAL, Vec3(0.7, 0.3, 0.3), Vec3(0, 0, 0), 0.1);
    
    // OpenCV image buffer
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Animation loop
    const int num_frames = 60;
    
    // Define cube vertices
    double cube_size = 2.0;
    Vec3 cube_center(0, 1.5, 0);
    double half = cube_size / 2.0;
    
    // Set block and grid dimensions for kernels
    int tx = 16;
    int ty = 16;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    
    // Initialize random number generator
    init_rand<<<blocks, threads>>>(d_rand_state, width, height, time(NULL));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Allocate device memory for scene
    GPUScene *d_scene;
    CUDA_CHECK(cudaMallocManaged(&d_scene, sizeof(GPUScene)));
    
    for (int frame = 0; frame < num_frames; ++frame) {
        std::cout << "Rendering frame " << frame + 1 << "/" << num_frames << "..." << std::endl;
        
        // Calculate cube vertices with rotation for this frame
        double angle = 2.0 * M_PI * frame / num_frames;
        double cos_angle = cos(angle);
        double sin_angle = sin(angle);
        
        // Define the 8 corners of the cube with rotation
        Vec3 p1 = Vec3(cube_center.x + (-half * cos_angle - (-half) * sin_angle), 
                      cube_center.y - half, 
                      cube_center.z + (-half * sin_angle + (-half) * cos_angle));
        
        Vec3 p2 = Vec3(cube_center.x + (half * cos_angle - (-half) * sin_angle), 
                      cube_center.y - half, 
                      cube_center.z + (half * sin_angle + (-half) * cos_angle));
        
        Vec3 p3 = Vec3(cube_center.x + (half * cos_angle - half * sin_angle), 
                      cube_center.y + half, 
                      cube_center.z + (half * sin_angle + half * cos_angle));
        
        Vec3 p4 = Vec3(cube_center.x + (-half * cos_angle - half * sin_angle), 
                      cube_center.y + half, 
                      cube_center.z + (-half * sin_angle + half * cos_angle));
        
        Vec3 p5 = Vec3(cube_center.x + (-half * cos_angle - (-half) * sin_angle), 
                      cube_center.y - half, 
                      cube_center.z + (-half * sin_angle + (-half) * cos_angle) + cube_size);
        
        Vec3 p6 = Vec3(cube_center.x + (half * cos_angle - (-half) * sin_angle), 
                      cube_center.y - half, 
                      cube_center.z + (half * sin_angle + (-half) * cos_angle) + cube_size);
        
        Vec3 p7 = Vec3(cube_center.x + (half * cos_angle - half * sin_angle), 
                      cube_center.y + half, 
                      cube_center.z + (half * sin_angle + half * cos_angle) + cube_size);
        
        Vec3 p8 = Vec3(cube_center.x + (-half * cos_angle - half * sin_angle), 
                      cube_center.y + half, 
                      cube_center.z + (-half * sin_angle + half * cos_angle) + cube_size);
        
        // Update cube vertices in scene
        cpu_scene.cube_vertices[0] = p1;
        cpu_scene.cube_vertices[1] = p2;
        cpu_scene.cube_vertices[2] = p3;
        cpu_scene.cube_vertices[3] = p4;
        cpu_scene.cube_vertices[4] = p5;
        cpu_scene.cube_vertices[5] = p6;
        cpu_scene.cube_vertices[6] = p7;
        cpu_scene.cube_vertices[7] = p8;
        
        // Copy scene to device
        *d_scene = cpu_scene;
        
        // Launch kernel to render the scene
        render_kernel<<<blocks, threads>>>(
            output_image, width, height, samples_per_pixel, max_depth,
            d_rand_state, *d_scene, camera_pos, lower_left_corner, horizontal, vertical
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy rendered output to OpenCV image
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                int pixel_index = j * width + i;
                Vec3 color = output_image[pixel_index];
                
                // Write to image buffer
                image.at<cv::Vec3b>(j, i) = cv::Vec3b(
                    static_cast<unsigned char>(255.99 * color.z),
                    static_cast<unsigned char>(255.99 * color.y),
                    static_cast<unsigned char>(255.99 * color.x)
                );
            }
        }
        
        // Display the rendered image
        cv::imshow("CUDA Path Tracer", image);
        
        // Save the frame
        std::string filename = "frame_" + std::to_string(frame) + ".png";
        cv::imwrite(filename, image);
        
        // Wait for key press (30ms) - adjust for animation speed
        cv::waitKey(30);
    }
    
    // Wait for a key press before closing
    std::cout << "Rendering complete. Press any key to exit." << std::endl;
    cv::waitKey(0);
    
    // Free CUDA memory
    CUDA_CHECK(cudaFree(output_image));
    CUDA_CHECK(cudaFree(d_rand_state));
    CUDA_CHECK(cudaFree(d_scene));
    
    return 0;
}
