#include <bridge.h>

#include <torch/torch.h>
#include <torch/script.h>

// #include <torch/script.h>
// #include <Aten/ATen.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <chrono>
#include <thread>

// #include <opencv2/opencv.hpp>


#define def_bridge_simple(Name) \
    extern "C" bridge_tensor_t Name(bridge_tensor_t input) { \
        auto t_input = bridge_to_torch(input); \
        auto t_output = torch::Name(t_input); \
        return torch_to_bridge(t_output); \
    }



int bridge_tensor_elements(bridge_tensor_t &bt) {
    int size = 1;
    for (int i = 0; i < bt.dim; ++i) {
        size *= bt.sizes[i];
    }
    return size;
}

size_t bridge_tensor_size(bridge_tensor_t &bt) {
    return sizeof(float32_t) * bridge_tensor_elements(bt);
}

void store_tensor(torch::Tensor &input, float32_t* dest) {
    float32_t * data = input.data_ptr<float32_t>();
    size_t bytes_size = sizeof(float32_t) * input.numel();
    // std::memmove(dest,data,bytes_size);
    std::memcpy(dest,data,bytes_size);
}

bridge_tensor_t torch_to_bridge(torch::Tensor &tensor) {
    bridge_tensor_t result;
    result.created_by_c = true;
    result.dim = tensor.dim();
    result.sizes = new int32_t[result.dim];
    for (int i = 0; i < result.dim; ++i) {
        result.sizes[i] = tensor.size(i);
    }
    result.data = new float32_t[bridge_tensor_elements(result)];
    store_tensor(tensor, result.data);
    return result;
}

torch::Tensor bridge_to_torch(bridge_tensor_t &bt) {
    std::vector<int64_t> sizes_vec(bt.sizes, bt.sizes + bt.dim);
    auto shape = torch::IntArrayRef(sizes_vec);
    return torch::from_blob(bt.data, shape, torch::kFloat);
}

extern "C" float32_t* unsafe(const float32_t* arr) {
    return const_cast<float32_t*>(arr);
}

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)),(std::istreambuf_iterator<char>()));
    input.close();
    return bytes;
}

extern "C" bridge_tensor_t load_tensor_from_file(const uint8_t* file_path) {
    // // Load the tensor from a file
    // torch::Tensor tensor;
    // torch::load(tensor,file_path);

    // std::cout << "Tensor loaded from file: " << tensor.sizes() << std::endl;

    std::string fp(reinterpret_cast<const char*>(file_path));
    std::vector<char> f = get_the_bytes(fp);
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor t = x.toTensor();
    return torch_to_bridge(t);
}

extern "C" bridge_tensor_t load_tensor_dict_from_file(const uint8_t* file_path,const uint8_t* tensor_key) {
    std::string fp(reinterpret_cast<const char*>(file_path));
    std::string tk(reinterpret_cast<const char*>(tensor_key));

    torch::jit::script::Module container = torch::jit::load(fp);
    torch::Tensor tensor = container.attr(tk).toTensor();

    return torch_to_bridge(tensor);

}

extern "C" bridge_tensor_t load_run_model(const uint8_t* model_path, bridge_tensor_t input) {
    auto t_input = bridge_to_torch(input);
    std::string mp(reinterpret_cast<const char*>(model_path));

    std::cout << "Loading model from path: " << mp << std::endl;
    std::cout.flush();


    torch::jit::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(mp);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "error loading the model\n" << e.msg();
        // std::system("pause");
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(t_input);

    auto output = module.forward(inputs).toTensor();

    std::cout << "Model output: " << output.sizes() << std::endl;
    return torch_to_bridge(output);
}

extern "C" bridge_tensor_t increment3(bridge_tensor_t arr) {
    auto t = bridge_to_torch(arr);
    // Increment the tensor
    auto incremented_tensor = t + 1;

    return torch_to_bridge(incremented_tensor);
}

extern "C" bridge_tensor_t convolve2d(
    bridge_tensor_t input,
    bridge_tensor_t kernel,
    bridge_tensor_t bias,
    int stride,
    int padding
) {
    auto t_input = bridge_to_torch(input);
    auto t_kernel = bridge_to_torch(kernel);
    auto t_bias = bridge_to_torch(bias);
    auto output = torch::conv2d(t_input, t_kernel, t_bias, stride, padding);
    return torch_to_bridge(output);
}

extern "C" bridge_tensor_t conv2d(
    bridge_tensor_t input,
    bridge_tensor_t kernel,
    bridge_tensor_t bias,
    int stride,
    int padding
) {
    auto t_input = bridge_to_torch(input);
    auto t_kernel = bridge_to_torch(kernel);
    auto t_bias = bridge_to_torch(bias);
    auto output = torch::conv2d(t_input, t_kernel, t_bias, stride, padding);
    return torch_to_bridge(output);
}

extern "C" bridge_tensor_t matmul(bridge_tensor_t a, bridge_tensor_t b) {
    auto t_a = bridge_to_torch(a);
    auto t_b = bridge_to_torch(b);

    // std::cout << "Input A shape: " << t_a.sizes() << std::endl;
    // std::cout << "Input B shape: " << t_b.sizes() << std::endl;
    // std::cout.flush();

    auto output = torch::matmul(t_a, t_b);

    // std::cout << "Input A shape: " << t_a.sizes() << std::endl;
    // std::cout << "Input B shape: " << t_b.sizes() << std::endl;
    // std::cout << "Input A: " << t_a.sum() << std::endl;
    // std::cout << "Input B: " << t_b.sum() << std::endl;
    // // std::cout << "Input B: " << t_b << std::endl;
    // std::cout << "Output shape: " << output.sizes() << std::endl;
    // std::cout << "Output sum: " << output.sum() << std::endl;
    // std::cout.flush();
    // printf("Hello from matmul!\n");
    return torch_to_bridge(output);

    // auto output_copy = output.clone();
    // std::cout << "Output copy shape: " << output_copy.sizes() << std::endl;
    // std::cout.flush();

    // auto bt = torch_to_bridge(output_copy);
    // std::cout << "Bridge tensor sizes: " << bt.sizes << std::endl;
    // std::cout << "Bridge tensor dim: " << bt.dim << std::endl;

    // std::cout.flush();

    // return bt;
}

extern "C" bridge_tensor_t max_pool2d(
    bridge_tensor_t input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    auto t_input = bridge_to_torch(input);
    auto output = torch::max_pool2d(t_input, kernel_size, stride, padding);
    return torch_to_bridge(output);
}

extern "C" bridge_tensor_t resize(
    bridge_tensor_t input,
    int height,
    int width
) {
    auto image = bridge_to_torch(input);

    // auto output = resize_tensor_last2(image, height, width);
    
    // at::Tensor output = at::upsample_bilinear2d(t_input.unsqueeze(0), {height, width}, false);
    if (image.dim() == 3) {
        auto output = torch::nn::functional::interpolate(
            image.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({ height, width }))
            .mode(torch::kBilinear)
            .align_corners(false)
        ).squeeze(0);
        return torch_to_bridge(output);
    } else if (image.dim() == 4) {
        auto output = torch::nn::functional::interpolate(
            image,
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({ height, width }))
            .mode(torch::kBilinear)
            .align_corners(false)
        );
        return torch_to_bridge(output);
    } else {
        std::cerr << "Unsupported tensor dimension: " << image.dim() << std::endl;
        std::cerr.flush();
        std::cout << "Unsupported tensor dimension: " << image.dim() << std::endl;
        std::cout.flush();
        return input; // Return the original tensor if the dimension is unsupported
    }
}

extern "C" bridge_tensor_t imagenet_normalize(bridge_tensor_t input) {
    auto t_input = bridge_to_torch(input);
    torch::Tensor image = t_input; //.to(torch::kFloat32);// / 255.0;

    static const std::vector<float> kMean{0.485, 0.456, 0.406};
    static const std::vector<float> kStd {0.229, 0.224, 0.225};
    auto opts = image.options();
    auto mean = torch::tensor(kMean).reshape({3, 1, 1});  // (3,1,1)
    auto std  = torch::tensor(kStd).reshape({3, 1, 1});

    if (image.dim() == 4) {
        mean = mean.unsqueeze(0); // (1,3,1,1)
        std = std.unsqueeze(0);
    }

    auto output = (image - mean) / std;
    return torch_to_bridge(output);
}


extern "C" bridge_tensor_t add_two_arrays(bridge_tensor_t a, bridge_tensor_t b) {
    torch::Tensor t_a = bridge_to_torch(a);
    torch::Tensor t_b = bridge_to_torch(b);

    torch::Tensor output = t_a + t_b;

    return torch_to_bridge(output);
}

extern "C" bridge_tensor_t nll_loss(
    bridge_tensor_t input, 
    bridge_tensor_t target, 
    bridge_tensor_t weight,
    int ignoreIndex,
    int reduction
) {
    // Convert bridge_tensor_t to torch::Tensor
    at::Tensor t_input = bridge_to_torch(input).to(torch::kFloat32);
    at::Tensor t_target = bridge_to_torch(target).to(torch::kLong);
    at::Tensor t_weight = bridge_to_torch(weight).to(torch::kFloat32);

    // Map reduction int to torch::Reduction enum
    torch::nn::functional::NLLLossFuncOptions::reduction_t reduction_enum;
    switch (reduction) {
        case 0: reduction_enum = torch::kNone; break;
        case 1: reduction_enum = torch::kMean; break;
        case 2: reduction_enum = torch::kSum; break;
        default: reduction_enum = torch::kMean; break;
    }

    at::Tensor output = torch::nn::functional::nll_loss(
        t_input,
        t_target,
        torch::nn::functional::NLLLossFuncOptions()
            .weight(t_weight)
            .ignore_index(ignoreIndex)
            .reduction(reduction_enum)
    );

    if (output.dim() == 0) {
        output = output.unsqueeze(0); // Makes scalar result 1D with 1 element
    }

    return torch_to_bridge(output);
}

// extern "C" bridge_tensor_t capture_webcam_bridge(int cam_index) {
//     torch::Tensor image = capture_webcam(cam_index);
//     return torch_to_bridge(image);
// }



// extern "C"


//  
// extern "C" bridge_tensor_t conv2d(
//     bridge_tensor_t input,
//     bridge_tensor_t kernel,
//     nil_scalar_tensor_t bias,
//     nil_scalar_tensor_t stride,
//     nil_scalar_tensor_t padding
// ) {
//     namespace F = torch::nn::functional;
//     F::conv2d(input, kernel, F::Conv2dFuncOptions().stride(1));
// }


extern "C" int baz(void) {
    printf("Hello from baz!\n");
    auto x = torch::randn({5, 3});
    return x.size(0);
}


extern "C" void wrHello(void) {
    printf("Hello from wrHello!\n");
}


extern "C" void wrHelloTorch(void) {
    printf("Hello from wrHelloTorch!\n");
    // auto t = torch::ones({2, 3});
    // std::cout << t << std::endl;
}




extern "C" void increment(float* arr, int* sizes, int dim, float* output) {
    // Convert sizes to std::vector<int64_t>
    std::vector<int64_t> sizes_vec(sizes, sizes + dim);
    auto shape = at::IntArrayRef(sizes_vec);
    auto t = torch::from_blob(arr, shape, torch::kFloat);

    // // Increment the tensor
    // auto incremented_tensor = t + 1;

    // // Store the incremented tensor in the output array
    // storeTensor(incremented_tensor, output);

    auto incremented_tensor = torch::from_blob(output, shape, torch::kFloat);
    incremented_tensor.copy_(t + 1);
}

extern "C" bridge_tensor_t increment2(float* arr, int* sizes, int dim) {
    // Convert sizes to std::vector<int64_t>
    std::vector<int64_t> sizes_vec(sizes, sizes + dim);
    auto shape = at::IntArrayRef(sizes_vec);
    auto t = torch::from_blob(arr, shape, torch::kFloat);

    // // Increment the tensor
    // auto incremented_tensor = t + 1;

    // // Store the incremented tensor in the output array
    // storeTensor(incremented_tensor, output);

    auto incremented_tensor = t + 1;

    return torch_to_bridge(incremented_tensor);
}


extern "C" float sumArray(float* arr, int* sizes, int dim) {
    // Convert sizes to std::vector<int64_t>

    printf("sumArray called with arr: %p, sizes: %p, dim: %d\n", arr, sizes, dim);

    std::vector<int64_t> sizes_vec(sizes, sizes + dim);
    std::cout << sizes_vec << std::endl;

    auto shape = at::IntArrayRef(sizes_vec);
    std::cout << shape << std::endl;

    auto t = torch::from_blob(arr, shape, torch::kFloat);
    std::cout << t << std::endl;

    return t.sum().item<float>();

    // return 0.0f;

    // float sum = 0.0f;
    // for (int i = 0; i < size; ++i) {
    //     sum += arr[i];
    // }
    // return sum;
    // const std::vector<int64_t> sizes_vec(sizes, dim);
    // auto shape = at::IntArrayRef(sizes_vec);

    // auto t = torch::from_blob(arr, shape, torch::kFloat);
    // return t.sum().item<float>();
}


// extern "C" void split_loop(int64_t idx, int64_t n) {
//     for (int i = 0; i < n; ++i) {
//         std::cout << "idx(" << idx << "," << n << ") = " << i << std::endl;
//         std::cout.flush();
//     }
// }

// extern "C" void split_loop_filler(int64_t n,int64_t* ret) {
//     for (int i = 0; i < n; ++i) {
//         *ret = i;
//         std::this_thread::sleep_for(std::chrono::seconds(0));
//     }
// }



// cv::VideoCapture open_camera(int cam_index) {
//     cv::VideoCapture cap(cam_index, cv::CAP_AVFOUNDATION);
//     if (!cap.isOpened()) {
//         std::cerr << "Could not open camera index " << cam_index << std::endl;
//         return cv::VideoCapture();
//     }
//     cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // minimal internal buffering
//     cap.set(cv::CAP_PROP_FPS, 60);       // request higher FPS if possible
//     return cap;
// }


extern "C" void show_webcam(void) {
    // cv::VideoCapture cap;
    // cap = open_camera(0);

    // cv::Mat frame_bgr;

    // while (true) {
    //     if (!cap.read(frame_bgr) || frame_bgr.empty()) {
    //         std::cerr << "[WARN] Empty frame, exiting" << std::endl;
    //         break;
    //     }

    //     cv::imshow("webcam", frame_bgr);

    //     if (cv::waitKey(1) == 27) { // ESC key
    //         break;
    //     }
    // }

    // cap.release();
    // cv::destroyAllWindows();
}