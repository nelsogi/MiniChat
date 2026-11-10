using MiniChat.App;
using MiniChat.UI;

namespace MiniChat.Commands;

/// <summary>
/// Exporte un modèle en exports/&lt;nom&gt;.nelmodel, à installer dans Nel Code avec /import (voir NELMODEL-FORMAT.md).
/// Par défaut le modèle de chat (model) ; --from model-pretrain (ou un nom de /save) pour un autre.
/// </summary>
public sealed class ExportCommand : ICommand
{
    public string Name => "export";
    public string Description => "Exporte un modèle en exports/<nom>.nelmodel, pour /import dans Nel Code";
    public string Usage => "/export <nom> [description] [--from model-pretrain]";

    public void Execute(CommandContext context, string[] args)
    {
        var words = args.ToList();
        string source = Workbench.ChatModel;
        int from = words.FindIndex(w => w.Equals("--from", StringComparison.OrdinalIgnoreCase));
        if (from >= 0)
        {
            if (from + 1 >= words.Count)
                throw new UserException($"--from attend un modèle (model, model-pretrain, ou un nom de /save). Utilisation : {Usage}");
            source = words[from + 1];
            words.RemoveRange(from, 2);
        }
        if (words.Count == 0)
            throw new UserException($"Quel nom ? Utilisation : {Usage}, par exemple : /export minichat-v2 Fine-tuné sur dialogues.txt");

        string name = words[0];
        // Les guillemets sont permis autour de la description : /export v2 "Mon modèle"
        string? description = words.Count > 1 ? string.Join(' ', words.Skip(1)).Trim('"', '«', '»', ' ') : null;

        var result = Messages.Spin($"Export de '{source}' et vérification...",
            () => context.Workbench.Export(name, source, string.IsNullOrEmpty(description) ? null : description));

        var info = result.Info;
        Messages.Success($"Modèle '{result.Source}' exporté dans {result.Path} ({result.Bytes / 1e6:0.0} Mo)" +
                         $"{(result.Replaced ? ", l'ancien export du même nom est remplacé" : "")}. Fichier relu : complet.");
        Messages.Dim($"{info.Parameters:N0} paramètres | {info.Architecture} | " +
                     $"loss de vérification {(info.ValLoss is { } loss ? loss.ToString("F3") : "?")} | " +
                     $"{(info.TrainingSteps is { } steps ? steps.ToString("N0") : "?")} pas" +
                     (info.Description is { } text ? $" | {text}" : ""));
        if (result.Source == Workbench.PretrainedModel)
            Messages.Dim("(modèle seulement pré-entraîné : il continue du texte, il ne sait pas encore discuter)");
    }
}
