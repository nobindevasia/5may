using System;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class FeatureSelectorFactory
    {
        private readonly MLContext _mlContext;

        public FeatureSelectorFactory(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        public IFeatureSelector CreateSelector(FeatureSelectionMethod method)
        {
            return method switch
            {
                FeatureSelectionMethod.None => new NoFeatureSelector(_mlContext),
                FeatureSelectionMethod.Correlation => new CorrelationFeatureSelector(_mlContext),
                FeatureSelectionMethod.PCA => new PCAFeatureSelector(_mlContext),

                _ => throw new ArgumentException($"Unsupported feature selection method: {method}")
            };
        }
    }
}