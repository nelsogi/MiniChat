using MiniChat.App;
using MiniChat.UI;

namespace MiniChat.Commands;

/// <summary>
/// Garde le modèle chargé sous un nom (models/saved/&lt;nom&gt;/). Utile avant un /train ou un /continue,
/// qui remplacent model/ et model-pretrain/ : l'ancien reste disponible pour /load et /compare.
/// </summary>
public sealed class SaveCommand : ICommand
{
    public string Name => "save";
    public string Description => "Garde le modèle chargé sous un nom, pour le recharger (/load) ou le comparer (/compare)";
    public string Usage => "/save <nom>";

    public void Execute(CommandContext context, string[] args)
    {
        if (args.Length != 1)
            throw new UserException($"Quel nom ? Utilisation : {Usage}, par exemple : /save avant-continue");

        string name = args[0];
        string source = context.Workbench.ModelSource ?? "?";
        bool replaced = Messages.Spin($"Sauvegarde de '{name}'...", () => context.Workbench.SaveAs(name));
        Messages.Success($"Modèle '{source}' gardé sous le nom '{name}'{(replaced ? " (l'ancien du même nom est remplacé)" : "")}.");
    }
}
