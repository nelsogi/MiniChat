using MiniChat.App;
using MiniChat.UI;

namespace MiniChat.Commands;

public sealed class LoadCommand : ICommand
{
    public string Name => "load";
    public string Description => "Charge un modèle sauvegardé (model, model-pretrain, ou un nom de /save) pour le chat";
    public string Usage => "/load <nom>";

    public void Execute(CommandContext context, string[] args)
    {
        if (args.Length != 1)
            throw new UserException($"Quel modèle ? Utilisation : {Usage} (/models pour la liste)");

        Messages.Spin($"Chargement de '{args[0]}'...", () => { context.Workbench.Load(args[0]); return 0; });
        var model = context.Workbench.Model!;
        Messages.Success($"Modèle '{context.Workbench.ModelSource}' chargé ({model.ParameterCount / 1e6:0.0} M paramètres). " +
                         "Nouvelle conversation.");
        if (context.Workbench.ModelSource == Workbench.PretrainedModel)
            Messages.Dim("(modèle seulement pré-entraîné : il continue du texte, il ne sait pas encore discuter)");
    }
}
