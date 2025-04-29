using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.DataBalancing
{
    public class SmoteDataBalancer : IDataBalancer
    {
        public async Task<IDataView> BalanceDataset(
            MLContext mlContext,
            IDataView data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField)
        {
            ValidateConfig(config);
            Console.WriteLine("=============== Balancing Dataset with SMOTE ===============");

            try
            {
                IDataView preparedData = data;
                if (data.Schema.GetColumnOrNull("Features") == null)
                {
                    var pipeline = mlContext.Transforms.Concatenate("Features", featureNames);
                    preparedData = pipeline.Fit(data).Transform(data);
                }

                var dataEnumerable = mlContext.Data.CreateEnumerable<FeatureVector>(
                    preparedData, reuseRowObject: false).ToList();

                var minorityClass = new List<float[]>();
                var majorityClass = new List<float[]>();
                var minorityLabels = new List<long>();
                var majorityLabels = new List<long>();

                foreach (var row in dataEnumerable)
                {
                    if (row.Label == 1)
                    {
                        minorityClass.Add(row.Features);
                        minorityLabels.Add(row.Label);
                    }
                    else
                    {
                        majorityClass.Add(row.Features);
                        majorityLabels.Add(row.Label);
                    }
                }

                if (minorityClass.Count > majorityClass.Count)
                {
                    var tempFeatures = minorityClass;
                    var tempLabels = minorityLabels;
                    minorityClass = majorityClass;
                    minorityLabels = majorityLabels;
                    majorityClass = tempFeatures;
                    majorityLabels = tempLabels;
                }

                Console.WriteLine($"Original counts - Minority: {minorityClass.Count}, Majority: {majorityClass.Count}");

                var random = new Random(42);
                int undersampledMajorityCount = (int)(majorityClass.Count * config.UndersamplingRatio);
                var shuffledIndices = Enumerable.Range(0, majorityClass.Count).ToList();
                for (int i = shuffledIndices.Count - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    var temp = shuffledIndices[i];
                    shuffledIndices[i] = shuffledIndices[j];
                    shuffledIndices[j] = temp;
                }

                var undersampledMajority = shuffledIndices
                    .Take(undersampledMajorityCount)
                    .Select(i => majorityClass[i])
                    .ToList();

                var undersampledMajorityLabels = shuffledIndices
                    .Take(undersampledMajorityCount)
                    .Select(i => majorityLabels[i])
                    .ToList();

                int targetMinorityCount = (int)(undersampledMajorityCount * config.MinorityToMajorityRatio);
                int syntheticCount = Math.Max(0, targetMinorityCount - minorityClass.Count);

                var syntheticSamples = await GenerateSyntheticSamples(
                    minorityClass,
                    syntheticCount,
                    config.KNeighbors,
                    random);

                var balancedFeatures = new List<FeatureVector>();

                balancedFeatures.AddRange(undersampledMajority.Select((f, i) => new FeatureVector
                {
                    Features = f,
                    Label = undersampledMajorityLabels[i]
                }));

                balancedFeatures.AddRange(minorityClass.Select((f, i) => new FeatureVector
                {
                    Features = f,
                    Label = minorityLabels[i]
                }));

                balancedFeatures.AddRange(syntheticSamples.Select(f => new FeatureVector
                {
                    Features = f,
                    Label = 1
                }));

                Console.WriteLine($"Final counts - Minority: {minorityClass.Count + syntheticSamples.Count}, " +
                                $"Majority: {undersampledMajority.Count}");

                return mlContext.Data.LoadFromEnumerable(balancedFeatures);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in SMOTE balancing: {ex.Message}");
                throw;
            }
        }

        private void ValidateConfig(DataBalancingConfig config)
        {
            if (config.UndersamplingRatio <= 0 || config.UndersamplingRatio > 1)
                throw new ArgumentException("Undersampling ratio must be between 0 and 1");

            if (config.MinorityToMajorityRatio <= 0 || config.MinorityToMajorityRatio > 1)
                throw new ArgumentException("Minority to majority ratio must be between 0 and 1");

            if (config.KNeighbors < 1)
                throw new ArgumentException("K should be greater than 0");
        }

        private async Task<List<float[]>> GenerateSyntheticSamples(
            List<float[]> minoritySamples,
            int syntheticCount,
            int k,
            Random random)
        {
            if (syntheticCount <= 0) return new List<float[]>();

            var synthetic = new List<float[]>();
            var samplesPerInstance = (int)Math.Ceiling((double)syntheticCount / minoritySamples.Count);

            await Task.Run(() =>
            {
                for (int i = 0; i < minoritySamples.Count && synthetic.Count < syntheticCount; i++)
                {
                    var neighbors = FindKNearestNeighbors(minoritySamples, minoritySamples[i], i, k);

                    for (int j = 0; j < samplesPerInstance && synthetic.Count < syntheticCount; j++)
                    {
                        var neighborIdx = random.Next(neighbors.Length);
                        var syntheticSample = InterpolateFeatures(
                            minoritySamples[i],
                            minoritySamples[neighbors[neighborIdx]],
                            random);
                        synthetic.Add(syntheticSample);
                    }
                }
            });

            return synthetic;
        }

        private int[] FindKNearestNeighbors(List<float[]> samples, float[] target, int excludeIndex, int k)
        {
            var distances = new List<(int index, float distance)>();

            for (int i = 0; i < samples.Count; i++)
            {
                if (i == excludeIndex) continue;
                distances.Add((i, EuclideanDistance(samples[i], target)));
            }

            return distances.OrderBy(x => x.distance)
                          .Take(k)
                          .Select(x => x.index)
                          .ToArray();
        }

        private float EuclideanDistance(float[] a, float[] b)
        {
            float sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return MathF.Sqrt(sum);
        }

        private float[] InterpolateFeatures(float[] a, float[] b, Random random)
        {
            float ratio = (float)random.NextDouble();
            var result = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + ratio * (b[i] - a[i]);
            }

            return result;
        }
        private class FeatureVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }
    }
}