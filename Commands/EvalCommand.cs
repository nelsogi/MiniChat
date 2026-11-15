using MiniChat.App;
using MiniChat.UI;

namespace MiniChat.Commands;

/// <summary>
/// L'évaluation notée : les 100 questions de data/eval.json, réponses gloutonnes, score par catégorie.
///   /eval          le modèle chargé
///   /eval &lt;nom&gt;    un modèle sauvegardé (/models pour la liste)
/// Chaque évaluation complète s'ajoute à l'historique du modèle (results/eval-&lt;nom&gt;.jsonl).
/// </summary>
public sealed class EvalCommand : ICommand
{
    public string Name => "eval";
    public string Description => "Évaluation notée (data/eval.json) : score par catégorie, historique par modèle";
    public string Usage => "/eval [nom]";

    public void Execute(CommandContext context, string[] args)
    {
        if (args.Length > 1)
            throw new UserException($"Un seul modèle à la fois. Utilisation : {Usage}");

        EvalView.Show(context.Workbench.Eval(args.Length == 1 ? args[0] : null, context.Cancellation));
    }
}
