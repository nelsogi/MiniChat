using System.Text;
using System.Text.Json;

namespace MiniChat.Evaluation;

/// <summary>
/// Une évaluation passée : quand, sur quel entraînement du modèle (RunId, pas, loss de vérification, voir
/// TrainingInfo ; null pour un modèle d'avant), et le score de chaque catégorie.
/// </summary>
public sealed record EvalRecord(DateTime Date, string Model, string? RunId, int? Step, float? ValLoss, int Correct, int Total,
    IReadOnlyList<EvalScore> Scores);

/// <summary>
/// L'historique des évaluations, un fichier par modèle : results/eval-&lt;nom&gt;.jsonl, une ligne JSON par /eval
/// (et par /compare). Le nom reste le même quand un nouvel entraînement remplace "model" : l'historique montre
/// alors le score de chaque entraînement à la suite. Seules les évaluations complètes y sont gardées.
/// </summary>
public sealed class EvalHistory
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = false };

    private readonly string _folder;

    public EvalHistory(string resultsFolder) => _folder = resultsFolder;

    public string PathOf(string model) => Path.Combine(_folder, $"eval-{model}.jsonl");

    public void Add(EvalRecord record)
    {
        Directory.CreateDirectory(_folder);
        File.AppendAllText(PathOf(record.Model), JsonSerializer.Serialize(record, JsonOptions) + "\n", new UTF8Encoding(false));
    }

    /// <summary>Les évaluations du modèle, de la plus ancienne à la plus récente. Ligne abîmée : ignorée.</summary>
    public IReadOnlyList<EvalRecord> Read(string model)
    {
        string path = PathOf(model);
        if (!File.Exists(path))
            return [];

        var records = new List<EvalRecord>();
        foreach (string line in File.ReadLines(path).Where(l => l.Trim().Length > 0))
        {
            try
            {
                if (JsonSerializer.Deserialize<EvalRecord>(line, JsonOptions) is { } record)
                    records.Add(record);
            }
            catch (JsonException)
            {
                // coupure pendant l'écriture : la ligne suivante est valable
            }
        }
        return records;
    }
}
