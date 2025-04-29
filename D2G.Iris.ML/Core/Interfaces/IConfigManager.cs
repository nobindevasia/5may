using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Enums;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IConfigManager
    {
        ModelConfig LoadConfiguration(string configPath);
        void ValidateConfiguration(ModelConfig config);
    }
}