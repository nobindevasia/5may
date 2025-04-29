using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.Configuration
{
    public class ConfigManager : IConfigManager
    {
        public ModelConfig LoadConfiguration(string configPath)
        {
            try
            {
                if (!File.Exists(configPath))
                {
                    throw new FileNotFoundException($"Configuration file not found at: {configPath}");
                }

                var configText = File.ReadAllText(configPath);
                Console.WriteLine("Reading config file...");

                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    WriteIndented = true,
                    Converters = { new JsonStringEnumConverter() }
                };

                var config = JsonSerializer.Deserialize<Dictionary<string, ModelConfig>>(configText, options);
                var modelConfig = config["modelConfig"];

                ValidateConfiguration(modelConfig);

                return modelConfig;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading configuration: {ex.Message}");
                throw;
            }
        }

        public void ValidateConfiguration(ModelConfig config)
        {
            if (config.Database == null)
                throw new ArgumentNullException(nameof(config.Database), "Database configuration is missing");

            if (string.IsNullOrEmpty(config.Database.TableName))
                throw new ArgumentException("Table name is missing in database configuration");

            if (config.TrainingParameters == null)
                throw new ArgumentNullException(nameof(config.TrainingParameters), "Training parameters are missing");

            if (string.IsNullOrEmpty(config.TrainingParameters.Algorithm))
                throw new ArgumentException("Algorithm name is missing in training parameters");

            ValidateFeatureEngineeringConfig(config.FeatureEngineering);
        }

        private void ValidateFeatureEngineeringConfig(FeatureEngineeringConfig config)
        {
            if (config == null) return;

            switch (config.Method)
            {
                case Core.Enums.FeatureSelectionMethod.PCA:
                    if (config.NumberOfComponents <= 0)
                        throw new ArgumentException("Number of components must be greater than 0 for PCA");
                    break;

                case Core.Enums.FeatureSelectionMethod.Correlation:
                    if (config.MulticollinearityThreshold <= 0 || config.MulticollinearityThreshold >= 1)
                        throw new ArgumentException("Multicollinearity threshold must be between 0 and 1 for Correlation Selection");
                    break;
            }
        }
    }
}