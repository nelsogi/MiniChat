using MiniChat.Training;
using MiniChat.UI;
using Spectre.Console;

namespace MiniChat.Commands;

public sealed class InfoCommand : ICommand
{
    public string Name => "info";
    public string Description => "Taille du modèle, GPU ou CPU, réglages de génération, conversation";
    public string Usage => "/info";

    public void Execute(CommandContext context, string[] args)
    {
        var workbench = context.Workbench;
        var grid = new Grid().AddColumn(new GridColumn().PadRight(2)).AddColumn();

        if (workbench.Model is { } model)
        {
            var config = model.Config;
            grid.AddRow(Label("Modèle"), new Text(
                $"{model.ParameterCount:N0} paramètres | {config.Layers} couches, {config.Heads} têtes, EmbedDim {config.EmbedDim} | " +
                $"mémoire {config.BlockSize} tokens | vocabulaire {config.VocabSize} tokens | dossier '{workbench.ModelSource}'"));
        }
        else
        {
            grid.AddRow(Label("Modèle"), new Text("aucun (/train pour en entraîner un)", Theme.Style(Theme.Warning)));
        }

        string memory = workbench.OnGpu && GpuMemory.Read() is var (used, total) ? $" | mémoire GPU {used:N0} / {total:N0} Mo" : "";
        grid.AddRow(Label("Calcul"), new Text($"{(workbench.OnGpu ? "GPU NVIDIA (CUDA)" : "CPU")}{memory}"));

        if (context.Chat is { } chat)
        {
            grid.AddRow(Label("Génération"), new Text(
                $"température {chat.Temperature} | top-k {(chat.TopK == 0 ? "désactivé" : chat.TopK)} | " +
                $"top-p {(chat.TopP >= 1 ? "désactivé" : chat.TopP)} | réponse max {chat.MaxReplyTokens} tokens | " +
                $"graine {(chat.Seed is null ? "aléatoire" : chat.Seed)}"));
            grid.AddRow(Label("Conversation"), new Text(
                $"{chat.TurnCount} tour(s), dont {chat.TurnsInLastPrompt} dans le dernier prompt (+ l'identité)"));
        }

        AnsiConsole.Write(grid);
    }

    private static Markup Label(string text) => new($"[{Theme.Dim}]{text}[/]");
}
