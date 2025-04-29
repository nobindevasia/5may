using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class CorrelationFeatureSelector : BaseFeatureSelector
    {
        public CorrelationFeatureSelector(MLContext mlContext)
            : base(mlContext)
        {
        }

        public override async Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            InitializeReport("Correlation-based");

            try
            {

                ValidateCorrelationConfiguration(config);

                var featureValues = new List<double[]>();
                foreach (var feature in candidateFeatures)
                {
                    featureValues.Add(GetColumnValues(data, feature));
                }

                var targetValues = GetColumnValues(data, targetField);

                var targetCorrelations = new Dictionary<string, double>();
                for (int i = 0; i < candidateFeatures.Length; i++)
                {
                    var correlation = Correlation.Pearson(featureValues[i], targetValues);
                    targetCorrelations[candidateFeatures[i]] = Math.Abs(correlation);
                }

                var correlationMatrix = Matrix<double>.Build.Dense(
                    candidateFeatures.Length,
                    candidateFeatures.Length
                );

                for (int i = 0; i < candidateFeatures.Length; i++)
                {
                    for (int j = 0; j < candidateFeatures.Length; j++)
                    {
                        correlationMatrix[i, j] = Correlation.Pearson(
                            featureValues[i],
                            featureValues[j]
                        );
                    }
                }

                var sortedFeatures = targetCorrelations
                    .OrderByDescending(x => x.Value)
                    .Select(x => x.Key)
                    .ToList();

                _report.AppendLine("\nFeatures Ranked by Target Correlation:");
                foreach (var feature in sortedFeatures)
                {
                    _report.AppendLine($"{feature,-40} | {targetCorrelations[feature]:F4}");
                }

                var selectedFeatures = SelectFeaturesWithMulticollinearityCheck(
                    sortedFeatures,
                    candidateFeatures,
                    correlationMatrix,
                    config,
                    targetCorrelations);

                AddFeatureSelectionSummary(
                    candidateFeatures.Length,
                    selectedFeatures.Count,
                    selectedFeatures.ToArray());

                _report.AppendLine($"Multicollinearity threshold: {config.MulticollinearityThreshold}");
                _report.AppendLine("\nSelected Features with Correlation Values:");
                foreach (var feature in selectedFeatures)
                {
                    _report.AppendLine($"- {feature} (correlation with target: {targetCorrelations[feature]:F4})");
                }
                var transformedData = CreateFeaturesColumn(data, selectedFeatures.ToArray());

                return (transformedData, selectedFeatures.ToArray(), _report.ToString());
            }
            catch (Exception ex)
            {
                AddErrorToReport(ex);
                throw;
            }
        }

        private List<string> SelectFeaturesWithMulticollinearityCheck(
            List<string> sortedFeatures,
            string[] candidateFeatures,
            Matrix<double> correlationMatrix,
            FeatureEngineeringConfig config,
            Dictionary<string, double> targetCorrelations)
        {
            var selectedFeatures = new List<string>();

            foreach (var feature in sortedFeatures)
            {
                if (selectedFeatures.Count >= config.MaxFeatures)
                    break;

                bool isHighlyCorrelated = false;
                foreach (var selectedFeature in selectedFeatures)
                {
                    var i1 = Array.IndexOf(candidateFeatures, feature);
                    var i2 = Array.IndexOf(candidateFeatures, selectedFeature);
                    if (Math.Abs(correlationMatrix[i1, i2]) > config.MulticollinearityThreshold)
                    {
                        isHighlyCorrelated = true;
                        break;
                    }
                }

                if (!isHighlyCorrelated)
                {
                    selectedFeatures.Add(feature);
                }
            }
            if (selectedFeatures.Count == 0 && sortedFeatures.Count > 0)
            {
                selectedFeatures.Add(sortedFeatures[0]);
            }

            return selectedFeatures;
        }

        private double[] GetColumnValues(IDataView dataView, string columnName)
        {
            var column = dataView.Schema.GetColumnOrNull(columnName);
            if (!column.HasValue)
                throw new ArgumentException($"Column '{columnName}' not found in data");

            var type = column.Value.Type;

            if (type is NumberDataViewType numType)
            {
                if (numType.RawType == typeof(float))
                    return dataView.GetColumn<float>(columnName).Select(v => (double)v).ToArray();
                else if (numType.RawType == typeof(double))
                    return dataView.GetColumn<double>(columnName).ToArray();
                else if (numType.RawType == typeof(int))
                    return dataView.GetColumn<int>(columnName).Select(v => (double)v).ToArray();
                else if (numType.RawType == typeof(long))
                    return dataView.GetColumn<long>(columnName).Select(v => (double)v).ToArray();
                else
                    return dataView.GetColumn<float>(columnName).Select(v => (double)v).ToArray();
            }
            else if (type is BooleanDataViewType)
            {
                return dataView.GetColumn<bool>(columnName).Select(v => v ? 1.0 : 0.0).ToArray();
            }

            throw new NotSupportedException($"Column type {type} is not supported for correlation analysis");
        }

        private void ValidateCorrelationConfiguration(FeatureEngineeringConfig config)
        {
            base.ValidateConfiguration(config);

            if (config.MulticollinearityThreshold <= 0 || config.MulticollinearityThreshold >= 1)
                throw new ArgumentException("Multicollinearity threshold must be between 0 and 1 for Correlation Selection");

            if (config.MaxFeatures <= 0)
                throw new ArgumentException("Max features must be greater than 0 for Correlation Selection");
        }
    }
}