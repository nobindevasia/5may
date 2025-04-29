using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class PCAFeatureSelector : BaseFeatureSelector
    {
        public PCAFeatureSelector(MLContext mlContext)
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
            InitializeReport("PCA");

            try
            {
                ValidatePcaConfiguration(config, candidateFeatures.Length);
                int numberOfComponents = config.NumberOfComponents;

                _report.AppendLine($"Applying PCA with {numberOfComponents} components");
                _report.AppendLine($"Original feature count: {candidateFeatures.Length}");

                var initialPipeline = mlContext.Transforms.Concatenate("FeaturesTemp", candidateFeatures);
                var initialData = initialPipeline.Fit(data).Transform(data);

                var normalizePipeline = mlContext.Transforms.NormalizeMinMax("FeaturesNormalized", "FeaturesTemp");
                var normalizedData = normalizePipeline.Fit(initialData).Transform(initialData);

                var pcaPipeline = mlContext.Transforms.ProjectToPrincipalComponents(
                    outputColumnName: "Features",
                    inputColumnName: "FeaturesNormalized",
                    rank: numberOfComponents);
                var pcaData = pcaPipeline.Fit(normalizedData).Transform(normalizedData);

                string[] pcaFeatureNames = Enumerable.Range(1, numberOfComponents)
                    .Select(i => $"PCA_Component_{i}")
                    .ToArray();

                _report.AppendLine("\nPCA transformation completed successfully.");
                _report.AppendLine("\nPCA Components:");
                foreach (var name in pcaFeatureNames)
                {
                    _report.AppendLine($"  - {name}");
                }

                AddFeatureSelectionSummary(
                    candidateFeatures.Length,
                    pcaFeatureNames.Length,
                    pcaFeatureNames);

                return (pcaData, pcaFeatureNames, _report.ToString());
            }
            catch (Exception ex)
            {
                AddErrorToReport(ex);
                throw;
            }
        }

        private void ValidatePcaConfiguration(FeatureEngineeringConfig config, int maxComponents)
        {
            base.ValidateConfiguration(config);

            if (config.NumberOfComponents <= 0 || config.NumberOfComponents > maxComponents)
            {
                _report.AppendLine($"Warning: Invalid number of components ({config.NumberOfComponents}). " +
                                  $"Using {Math.Min(maxComponents, 3)} instead.");
                config.NumberOfComponents = Math.Min(maxComponents, 3);
            }
        }
    }
}