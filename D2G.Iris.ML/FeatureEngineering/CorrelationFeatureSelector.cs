using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using MathNet.Numerics.Statistics;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class CorrelationFeatureSelector : IFeatureSelector
    {
        private readonly MLContext _mlContext;
        private readonly StringBuilder _report;

        public CorrelationFeatureSelector(MLContext mlContext)
        {
            _mlContext = mlContext;
            _report = new StringBuilder();
        }

        private class FeatureRow
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }

        public async Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            _report.Clear();
            _report.AppendLine("\nCorrelation-based Feature Selection Results:");
            _report.AppendLine("----------------------------------------------");

            try
            {
                var rows = mlContext.Data.CreateEnumerable<FeatureRow>(
                    data, reuseRowObject: false).ToList();

                if (!rows.Any() || rows[0].Features == null)
                {
                    throw new InvalidOperationException("No valid feature data found");
                }

                var targetCorrelations = new Dictionary<string, double>();
                var targetValues = rows.Select(r => (double)r.Label).ToArray();

                for (int i = 0; i < candidateFeatures.Length; i++)
                {
                    var featureValues = rows.Select(r => (double)r.Features[i]).ToArray();
                    var correlation = Math.Abs(Correlation.Pearson(featureValues, targetValues));
                    targetCorrelations[candidateFeatures[i]] = correlation;
                }

                var sortedFeatures = targetCorrelations
                    .OrderByDescending(x => x.Value)
                    .ToList();

                _report.AppendLine("\nFeatures Ranked by Target Correlation:");
                foreach (var pair in sortedFeatures)
                {
                    _report.AppendLine($"{pair.Key,-40} | {pair.Value:F4}");
                }

                var selectedFeatures = new List<string>();
                var selectedIndices = new List<int>();

                foreach (var pair in sortedFeatures)
                {
                    if (selectedFeatures.Count >= config.MaxFeatures)
                        break;

                    var currentIndex = Array.IndexOf(candidateFeatures, pair.Key);
                    var currentValues = rows.Select(r => (double)r.Features[currentIndex]).ToArray();

                    bool isHighlyCorrelated = false;
                    foreach (var selectedIndex in selectedIndices)
                    {
                        var selectedValues = rows.Select(r => (double)r.Features[selectedIndex]).ToArray();
                        var correlation = Math.Abs(Correlation.Pearson(currentValues, selectedValues));

                        if (correlation > config.MulticollinearityThreshold)
                        {
                            isHighlyCorrelated = true;
                            break;
                        }
                    }

                    if (!isHighlyCorrelated)
                    {
                        selectedFeatures.Add(pair.Key);
                        selectedIndices.Add(currentIndex);
                    }
                }

                _report.AppendLine($"\nSelection Summary:");
                _report.AppendLine($"Original features: {candidateFeatures.Length}");
                _report.AppendLine($"Selected features: {selectedFeatures.Count}");
                _report.AppendLine($"Multicollinearity threshold: {config.MulticollinearityThreshold}");
                _report.AppendLine("\nSelected Features:");
                foreach (var feature in selectedFeatures)
                {
                    _report.AppendLine($"- {feature} (correlation with target: {targetCorrelations[feature]:F4})");
                }

                var selectedRows = rows.Select(row => new FeatureRow
                {
                    Features = selectedIndices.Select(i => row.Features[i]).ToArray(),
                    Label = row.Label
                }).ToList();

                var transformedData = mlContext.Data.LoadFromEnumerable(selectedRows);

                return (transformedData, selectedFeatures.ToArray(), _report.ToString());
            }
            catch (Exception ex)
            {
                _report.AppendLine($"Error during correlation analysis: {ex.Message}");
                Console.WriteLine($"Full error details: {ex}");
                throw;
            }
        }
    }
}