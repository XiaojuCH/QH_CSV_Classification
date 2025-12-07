using System;
using System.Runtime.InteropServices;
using System.Text;

namespace ClassifierDemo
{
    public class Classifier : IDisposable
    {
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
        private static extern void Cleanup();

        private bool _initialized = false;

        public class PredictionResult
        {
            public int ClassIndex { get; set; }
            public string ClassName { get; set; } = "";
            public float Confidence { get; set; }
            public float[] Probabilities { get; set; } = new float[6];
        }

        public class BatchPredictionResult
        {
            public int SampleCount { get; set; }
            public PredictionResult[] Results { get; set; } = Array.Empty<PredictionResult>();
        }

        public bool InitializeClassifier(string modelPath, string scalerPath, string labelPath)
        {
            int result = Initialize(modelPath, scalerPath, labelPath);
            if (result == 0)
            {
                _initialized = true;
                return true;
            }

            string errorMessage = result switch
            {
                -1 => "无法打开标准化参数文件",
                -2 => "标准化参数无效",
                -3 => "类别映射无效",
                _ => $"初始化失败，错误代码: {result}"
            };
            throw new Exception(errorMessage);
        }

        public PredictionResult Predict(float[] features)
        {
            if (!_initialized)
                throw new InvalidOperationException("分类器未初始化");

            if (features.Length != 20)
                throw new ArgumentException("特征数组必须包含20个元素");

            float[] probabilities = new float[6];
            int predictedClass;

            int result = Predict(features, 20, probabilities, out predictedClass);

            if (result != 0)
                throw new Exception($"预测失败，错误代码: {result}");

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

        public BatchPredictionResult PredictFromCSV(string csvPath)
        {
            if (!_initialized)
                throw new InvalidOperationException("分类器未初始化");

            int maxSamples = 1000;
            int[] predictedClasses = new int[maxSamples];
            float[] allProbabilities = new float[maxSamples * 6];
            int sampleCount;

            int result = PredictFromCSV(csvPath, predictedClasses, allProbabilities, out sampleCount);

            if (result != 0)
                throw new Exception($"批量预测失败，错误代码: {result}");

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
