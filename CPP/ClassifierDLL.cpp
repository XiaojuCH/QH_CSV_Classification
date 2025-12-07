/**
 * LightGBM Classifier DLL
 * For C# WPF Integration
 * Supports both single prediction and batch CSV processing
 */

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <memory>

using namespace std;

// Global variables for model session
static unique_ptr<Ort::Env> g_env = nullptr;
static unique_ptr<Ort::Session> g_session = nullptr;
static vector<double> g_mean;
static vector<double> g_scale;
static vector<string> g_labels;
static bool g_initialized = false;

// Helper function: Parse JSON array
vector<double> parse_json_array(const string& json_content, const string& key) {
    vector<double> result;
    size_t key_pos = json_content.find("\"" + key + "\"");
    if (key_pos == string::npos) return result;

    size_t array_start = json_content.find("[", key_pos);
    size_t array_end = json_content.find("]", array_start);
    string array_str = json_content.substr(array_start + 1, array_end - array_start - 1);

    stringstream ss(array_str);
    string item;
    while (getline(ss, item, ',')) {
        item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) {
            result.push_back(stod(item));
        }
    }
    return result;
}

// Helper function: Load labels
vector<string> load_labels(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        return {};
    }

    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    vector<string> labels(6);

    for (int i = 0; i < 6; i++) {
        string key = "\"" + to_string(i) + "\"";
        size_t pos = content.find(key);
        if (pos != string::npos) {
            size_t value_start = content.find("\"", pos + key.length()) + 1;
            size_t value_end = content.find("\"", value_start);
            labels[i] = content.substr(value_start, value_end - value_start);
        }
    }
    return labels;
}

// Helper function: Read CSV file
vector<vector<float>> read_csv(const string& filename) {
    vector<vector<float>> data;
    ifstream file(filename);

    if (!file.is_open()) {
        return data;
    }

    string line;
    int line_number = 0;

    while (getline(file, line)) {
        line_number++;
        if (line.empty()) continue;

        vector<float> row;
        stringstream ss(line);
        string value;

        while (getline(ss, value, ',')) {
            try {
                row.push_back(stof(value));
            } catch (const exception& e) {
                break;
            }
        }

        if (row.size() == 20) {
            data.push_back(row);
        }
    }

    return data;
}

// DLL Export: Initialize the classifier
extern "C" __declspec(dllexport) int Initialize(const wchar_t* model_path,
                                                 const wchar_t* scaler_path,
                                                 const wchar_t* label_path) {
    try {
        // Load scaler parameters - use wifstream for wide char paths
        wifstream scaler_file(scaler_path);
        if (!scaler_file.is_open()) {
            return -1; // Failed to open scaler file
        }

        // Read file content
        wstring scaler_content_w((istreambuf_iterator<wchar_t>(scaler_file)), istreambuf_iterator<wchar_t>());

        // Convert to narrow string for parsing
        string scaler_content(scaler_content_w.begin(), scaler_content_w.end());

        g_mean = parse_json_array(scaler_content, "mean");
        g_scale = parse_json_array(scaler_content, "scale");

        if (g_mean.size() != 20 || g_scale.size() != 20) {
            return -2; // Invalid scaler parameters
        }

        // Load labels - use wifstream for wide char paths
        wifstream label_file(label_path);
        if (!label_file.is_open()) {
            return -3; // Failed to open label file
        }

        wstring label_content_w((istreambuf_iterator<wchar_t>(label_file)), istreambuf_iterator<wchar_t>());
        string label_content(label_content_w.begin(), label_content_w.end());

        // Parse labels
        g_labels.resize(6);
        for (int i = 0; i < 6; i++) {
            string key = "\"" + to_string(i) + "\"";
            size_t pos = label_content.find(key);
            if (pos != string::npos) {
                size_t value_start = label_content.find("\"", pos + key.length()) + 1;
                size_t value_end = label_content.find("\"", value_start);
                g_labels[i] = label_content.substr(value_start, value_end - value_start);
            }
        }

        if (g_labels.size() != 6) {
            return -3; // Invalid labels
        }

        // Initialize ONNX Runtime
        g_env = make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "ClassifierDLL");

        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        session_options.SetIntraOpNumThreads(1);

        g_session = make_unique<Ort::Session>(*g_env, model_path, session_options);

        g_initialized = true;
        return 0; // Success

    } catch (const exception& e) {
        return -999; // Unknown error
    }
}

// DLL Export: Predict single sample
extern "C" __declspec(dllexport) int Predict(const float* features,
                                              int feature_count,
                                              float* probabilities,
                                              int* predicted_class) {
    if (!g_initialized) {
        return -1; // Not initialized
    }

    if (feature_count != 20) {
        return -2; // Invalid feature count
    }

    try {
        // Standardize features
        vector<float> scaled_features(20);
        for (int i = 0; i < 20; i++) {
            scaled_features[i] = (features[i] - g_mean[i]) / g_scale[i];
        }

        // Create input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        vector<int64_t> input_shape = {1, 20};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            scaled_features.data(),
            scaled_features.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = g_session->GetInputNameAllocated(0, allocator);
        auto output_name_ptr = g_session->GetOutputNameAllocated(1, allocator); // Index 1 for probabilities

        const char* input_names[] = {input_name_ptr.get()};
        const char* output_names[] = {output_name_ptr.get()};

        // Run inference
        auto output_tensors = g_session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // Get probabilities
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        // Copy probabilities to output
        int max_idx = 0;
        float max_prob = output_data[0];

        for (int i = 0; i < 6; i++) {
            probabilities[i] = output_data[i];
            if (output_data[i] > max_prob) {
                max_prob = output_data[i];
                max_idx = i;
            }
        }

        *predicted_class = max_idx;

        return 0; // Success

    } catch (const exception& e) {
        return -999; // Unknown error
    }
}

// DLL Export: Predict from CSV file (batch processing)
extern "C" __declspec(dllexport) int PredictFromCSV(const wchar_t* csv_path,
                                                     int* predicted_classes,
                                                     float* all_probabilities,
                                                     int* sample_count) {
    if (!g_initialized) {
        return -1; // Not initialized
    }

    try {
        // Read CSV file using wifstream for wide char paths
        wifstream file(csv_path);
        if (!file.is_open()) {
            *sample_count = 0;
            return -2; // Cannot open file
        }

        vector<vector<float>> samples;
        wstring line;
        int line_number = 0;

        while (getline(file, line)) {
            line_number++;
            if (line.empty()) continue;

            vector<float> row;
            wstringstream ss(line);
            wstring value;

            while (getline(ss, value, L',')) {
                try {
                    row.push_back(stof(value));
                } catch (const exception& e) {
                    break;
                }
            }

            if (row.size() == 20) {
                samples.push_back(row);
            }
        }

        if (samples.empty()) {
            *sample_count = 0;
            return -2; // No valid data
        }

        *sample_count = samples.size();

        // Predict each sample
        for (size_t i = 0; i < samples.size(); i++) {
            float probabilities[6];
            int predicted_class;

            int result = Predict(samples[i].data(), 20, probabilities, &predicted_class);

            if (result != 0) {
                return result; // Prediction failed
            }

            // Store results
            predicted_classes[i] = predicted_class;
            for (int j = 0; j < 6; j++) {
                all_probabilities[i * 6 + j] = probabilities[j];
            }
        }

        return 0; // Success

    } catch (const exception& e) {
        return -999; // Unknown error
    }
}

// DLL Export: Get class name by index
extern "C" __declspec(dllexport) int GetClassName(int class_index, wchar_t* buffer, int buffer_size) {
    if (!g_initialized) {
        return -1; // Not initialized
    }

    if (class_index < 0 || class_index >= 6) {
        return -2; // Invalid class index
    }

    const string& label = g_labels[class_index];
    mbstowcs(buffer, label.c_str(), buffer_size);

    return 0; // Success
}

// DLL Export: Get number of classes
extern "C" __declspec(dllexport) int GetClassCount() {
    return 6;
}

// DLL Export: Get feature count
extern "C" __declspec(dllexport) int GetFeatureCount() {
    return 20;
}

// DLL Export: Cleanup resources
extern "C" __declspec(dllexport) void Cleanup() {
    g_session.reset();
    g_env.reset();
    g_mean.clear();
    g_scale.clear();
    g_labels.clear();
    g_initialized = false;
}
