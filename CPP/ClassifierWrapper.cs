using System;
using System.Runtime.InteropServices;
using System.Text;

namespace OpticalClassifier
{
    /// <summary>
    /// LightGBM 分类器 DLL 包装类
    /// 支持单样本预测和批量 CSV 处理
    /// </summary>
    public class Classifier : IDisposable
    {
        // DLL 导入声明
        [DllImport("ClassifierDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int Initialize(
            [MarshalAs(UnmanagedType.LPWStr)] string modelPath,
            [MarshalAs(UnmanagedType.LPWStr)] string scalerPath,
            [MarshalAs(UnmanagedType.LPWStr)] string labelPath);

        [DllImport("ClassifierDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int Predict(
            float[] features,
            int featureCount,
            float[] probabilities,
            out int predictedClass);

        [DllImport("ClassifierDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int PredictFromCSV(
            [MarshalAs(UnmanagedType.LPWStr)] string csvPath,
            int[] predictedClasses,
            float[] allProbabilities,
            out int sampleCount);

        [DllImport("ClassifierDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GetClassName(
            int classIndex,
            [MarshalAs(UnmanagedType.LPWStr)] StringBuilder buffer,
            int bufferSize);

        [DllImport("ClassifierDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GetClassCount();

        [DllImport("ClassifierDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GetFeatureCount();

        [DllImport("ClassifierDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Cleanup();

        private bool _initialized = false;

        /// <summary>
        /// 初始化分类器
        /// </summary>
        /// <param name="modelPath">ONNX 模型文件路径</param>
        /// <param name="scalerPath">标准化参数文件路径</param>
        /// <param name="labelPath">类别映射文件路径</param>
        /// <returns>是否初始化成功</returns>
        public bool Initialize(string modelPath, string scalerPath, string labelPath)
        {
            int result = Initialize(modelPath, scalerPath, labelPath);

            if (result == 0)
            {
                _initialized = true;
                return true;
            }

            // 错误处理
            string errorMessage = result switch
            {
                -1 => "无法打开标准化参数文件",
                -2 => "标准化参数无效（需要20个特征）",
                -3 => "类别映射无效（需要6个类别）",
                -999 => "未知错误",
                _ => $"初始化失败，错误代码: {result}"
            };

            throw new Exception(errorMessage);
        }

        /// <summary>
        /// 预测结果类
        /// </summary>
        public class PredictionResult
        {
            /// <summary>预测的类别索引 (0-5)</summary>
            public int ClassIndex { get; set; }

            /// <summary>预测的类别名称</summary>
            public string ClassName { get; set; }

            /// <summary>置信度 (0-1)</summary>
            public float Confidence { get; set; }

            /// <summary>所有类别的概率分布</summary>
            public float[] Probabilities { get; set; }
        }

        /// <summary>
        /// 批量预测结果类
        /// </summary>
        public class BatchPredictionResult
        {
            /// <summary>样本数量</summary>
            public int SampleCount { get; set; }

            /// <summary>每个样本的预测结果</summary>
            public PredictionResult[] Results { get; set; }
        }

        /// <summary>
        /// 预测单个样本
        /// </summary>
        /// <param name="features">20个特征值</param>
        /// <returns>预测结果</returns>
        public PredictionResult Predict(float[] features)
        {
            if (!_initialized)
            {
                throw new InvalidOperationException("分类器未初始化，请先调用 Initialize()");
            }

            if (features == null || features.Length != 20)
            {
                throw new ArgumentException("特征数组必须包含20个元素");
            }

            float[] probabilities = new float[6];
            int predictedClass;

            int result = Predict(features, 20, probabilities, out predictedClass);

            if (result != 0)
            {
                string errorMessage = result switch
                {
                    -1 => "分类器未初始化",
                    -2 => "特征数量无效",
                    -999 => "预测过程中发生未知错误",
                    _ => $"预测失败，错误代码: {result}"
                };
                throw new Exception(errorMessage);
            }

            // 获取类别名称
            StringBuilder className = new StringBuilder(256);
            GetClassName(predictedClass, className, 256);

            return new PredictionResult
            {
                ClassIndex = predictedClass,
                ClassName = className.ToString(),
                Confidence = probabilities[predictedClass],
                Probabilities = probabilities
            };
        }

        /// <summary>
        /// 从 CSV 文件批量预测
        /// </summary>
        /// <param name="csvPath">CSV 文件路径</param>
        /// <returns>批量预测结果</returns>
        public BatchPredictionResult PredictFromCSV(string csvPath)
        {
            if (!_initialized)
            {
                throw new InvalidOperationException("分类器未初始化，请先调用 Initialize()");
            }

            if (string.IsNullOrEmpty(csvPath))
            {
                throw new ArgumentException("CSV 文件路径不能为空");
            }

            // 预分配最大可能的数组（假设最多1000个样本）
            int maxSamples = 1000;
            int[] predictedClasses = new int[maxSamples];
            float[] allProbabilities = new float[maxSamples * 6];
            int sampleCount;

            int result = PredictFromCSV(csvPath, predictedClasses, allProbabilities, out sampleCount);

            if (result != 0)
            {
                string errorMessage = result switch
                {
                    -1 => "分类器未初始化",
                    -2 => "CSV 文件为空或无有效数据",
                    -999 => "预测过程中发生未知错误",
                    _ => $"批量预测失败，错误代码: {result}"
                };
                throw new Exception(errorMessage);
            }

            // 构建结果
            var results = new PredictionResult[sampleCount];

            for (int i = 0; i < sampleCount; i++)
            {
                int classIndex = predictedClasses[i];
                float[] probabilities = new float[6];

                for (int j = 0; j < 6; j++)
                {
                    probabilities[j] = allProbabilities[i * 6 + j];
                }

                StringBuilder className = new StringBuilder(256);
                GetClassName(classIndex, className, 256);

                results[i] = new PredictionResult
                {
                    ClassIndex = classIndex,
                    ClassName = className.ToString(),
                    Confidence = probabilities[classIndex],
                    Probabilities = probabilities
                };
            }

            return new BatchPredictionResult
            {
                SampleCount = sampleCount,
                Results = results
            };
        }

        /// <summary>
        /// 获取类别名称
        /// </summary>
        /// <param name="classIndex">类别索引 (0-5)</param>
        /// <returns>类别名称</returns>
        public string GetClassName(int classIndex)
        {
            if (!_initialized)
            {
                throw new InvalidOperationException("分类器未初始化");
            }

            if (classIndex < 0 || classIndex >= 6)
            {
                throw new ArgumentOutOfRangeException(nameof(classIndex), "类别索引必须在 0-5 之间");
            }

            StringBuilder buffer = new StringBuilder(256);
            int result = GetClassName(classIndex, buffer, 256);

            if (result != 0)
            {
                throw new Exception($"获取类别名称失败，错误代码: {result}");
            }

            return buffer.ToString();
        }

        /// <summary>
        /// 获取类别数量
        /// </summary>
        /// <returns>类别数量（固定为6）</returns>
        public int GetClassCount()
        {
            return GetClassCount();
        }

        /// <summary>
        /// 获取特征数量
        /// </summary>
        /// <returns>特征数量（固定为20）</returns>
        public int GetFeatureCount()
        {
            return GetFeatureCount();
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            if (_initialized)
            {
                Cleanup();
                _initialized = false;
            }
        }
    }
}
