using Microsoft.ML.Data;

namespace MLTest
{
    public class ModelInput
    {
        [ColumnName("isMale"), LoadColumn(0)]
        public bool IsMale { get; set; }

        [ColumnName("age"), LoadColumn(1)]
        public float Age { get; set; }

        [ColumnName("job"), LoadColumn(2)]
        public string Job { get; set; }

        [ColumnName("income"), LoadColumn(3)]
        public float Income { get; set; }

        [ColumnName("satisfac"), LoadColumn(4)]
        public string Satisfac { get; set; }
    }
}
