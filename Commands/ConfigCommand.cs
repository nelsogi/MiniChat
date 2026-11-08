using MiniChat.App;
using MiniChat.UI;
using Spectre.Console;

namespace MiniChat.Commands;

/// <summary>
/// Les réglages de settings.json :
///   /config                          les affiche
///   /config temperature 0.7          change un réglage, le sauvegarde, et l'applique à la conversation en cours
///   /config defaultModel model-pretrain   modèle chargé au prochain démarrage
/// </summary>
public sealed class ConfigCommand : ICommand
{
    public string Name => "config";
    public string Description => "Affiche ou modifie les réglages enregistrés dans settings.json";
    public string Usage => "/config [temperature|topK|topP|defaultModel <valeur>]";

    public void Execute(CommandContext context, string[] args)
    {
        var settings = context.Settings;
        if (args.Length == 0)
        {
            Show(context);
            return;
        }
        if (args.Length != 2)
            throw new UserException($"Utilisation : {Usage}");

        string value = args[1];
        var chat = context.Chat;
        switch (args[0].ToLowerInvariant())
        {
            case "temperature" or "temp":
                settings.Temperature = Settings.ClampTemperature(CommandContext.ParseNumber(value, Usage));
                if (chat is not null) chat.Temperature = settings.Temperature;
                break;
            case "topk":
                settings.TopK = Settings.ClampTopK((int)CommandContext.ParseNumber(value, Usage));
                if (chat is not null) chat.TopK = settings.TopK;
                break;
            case "topp":
                settings.TopP = Settings.ClampTopP(CommandContext.ParseNumber(value, Usage));
                if (chat is not null) chat.TopP = settings.TopP;
                break;
            case "defaultmodel" or "model":
                settings.DefaultModel = Settings.Models.FirstOrDefault(m => m.Equals(value, StringComparison.OrdinalIgnoreCase))
                    ?? throw new UserException($"Modèle inconnu : « {value} ». Choix : {string.Join(", ", Settings.Models)}.");
                context.Workbench.DefaultModel = settings.DefaultModel;
                break;
            default:
                throw new UserException($"Réglage inconnu : « {args[0]} ». Réglages : temperature, topK, topP, defaultModel.");
        }

        settings.Save(context.SettingsPath);
        Messages.Success($"Enregistré dans {Settings.FileName}.");
        if (args[0].ToLowerInvariant() is "defaultmodel" or "model" && context.Workbench.ModelSource != settings.DefaultModel)
            Messages.Dim($"(chargé au prochain démarrage ; le chat garde '{context.Workbench.ModelSource ?? "aucun"}' pour l'instant)");
        Show(context);
    }

    private static void Show(CommandContext context)
    {
        var settings = context.Settings;
        var table = new Table()
            .Border(TableBorder.Simple)
            .BorderStyle(Theme.Style(Theme.Border))
            .AddColumn(new TableColumn($"[{Theme.Dim}]Réglage[/]").NoWrap())
            .AddColumn(new TableColumn($"[{Theme.Dim}]Valeur[/]").NoWrap())
            .AddColumn(new TableColumn($"[{Theme.Dim}]Ce qu'il fait[/]"));

        Row(table, "temperature", settings.Temperature.ToString(), "plus bas = réponses sages, plus haut = créatives (0.1 à 2)");
        Row(table, "topK", settings.TopK == 0 ? "0 (désactivé)" : settings.TopK.ToString(), "tirage parmi les K tokens les plus probables");
        Row(table, "topP", settings.TopP >= 1 ? "1 (désactivé)" : settings.TopP.ToString(), "tirage parmi les tokens qui font cette part des chances");
        Row(table, "defaultModel", settings.DefaultModel, $"modèle chargé au démarrage : {string.Join(" ou ", Settings.Models)}");

        AnsiConsole.Write(table);
        Messages.Dim($"Fichier : {context.SettingsPath}");
        if (context.Chat is { } chat && (chat.Temperature != settings.Temperature || chat.TopK != settings.TopK || chat.TopP != settings.TopP))
            Messages.Dim($"Conversation en cours (/temp, /topk, /topp) : température {chat.Temperature}, top-k {chat.TopK}, top-p {chat.TopP}.");
    }

    private static void Row(Table table, string name, string value, string description) =>
        table.AddRow(new Text(name, Theme.Style(Theme.Command)), new Text(value), new Text(description, Theme.Style(Theme.Dim)));
}
