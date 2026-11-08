using System.Collections.Concurrent;
using MiniChat.Chat;
using Spectre.Console;

namespace MiniChat.UI;

/// <summary>
/// La conversation à l'écran :
///   > Bonjour !                    <- le message de l'utilisateur
///   ● Bonjour, comment vas-tu ?    <- la réponse, écrite au fur et à mesure
/// Un spinner tourne tant que le modèle n'a pas écrit son premier caractère.
/// </summary>
public static class ChatView
{
    public static void ShowUser(string message)
    {
        AnsiConsole.MarkupLine($"[{Theme.Dim}]{Theme.Prompt}[/] [{Theme.User}]{Markup.Escape(message)}[/]");
    }

    /// <summary>
    /// Envoie message au modèle et affiche sa réponse caractère par caractère.
    /// cancellation (Échap) : la réponse s'arrête là, "(interrompu)" est ajouté.
    /// </summary>
    public static void Reply(ChatSession chat, string message, CancellationToken cancellation = default)
    {
        // La génération tourne à part et pousse ses caractères ici ; l'affichage les reprend dans l'ordre.
        // Comme ça le spinner (sur ce thread) peut tourner jusqu'au premier caractère, puis laisser la place.
        using var characters = new BlockingCollection<char>();
        var generation = Task.Run(() =>
        {
            try { chat.Send(message, characters.Add, cancellation); }
            finally { characters.CompleteAdding(); }
        });

        char first = default;
        bool any = false;
        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .SpinnerStyle(Theme.Style(Theme.Accent))
            .Start($"[{Theme.Dim}]Réflexion... (Échap pour arrêter)[/]",
                _ => any = characters.TryTake(out first, Timeout.Infinite));

        AnsiConsole.Markup($"[{Theme.Accent}]{Theme.ReplyBullet}[/] ");
        if (any)
        {
            Write(first);
            foreach (char c in characters.GetConsumingEnumerable())
                Write(c);
        }
        else if (!cancellation.IsCancellationRequested)
        {
            AnsiConsole.Markup($"[{Theme.Dim}](pas de réponse)[/]");
        }
        if (cancellation.IsCancellationRequested)
            AnsiConsole.Markup($"[{Theme.Dim}] (interrompu)[/]");
        AnsiConsole.WriteLine();

        generation.GetAwaiter().GetResult();   // une erreur pendant la génération remonte ici

        // La vitesse compte tout, prompt compris : c'est l'attente réelle.
        if (chat.Generator.LastStats is { Tokens: > 0 } stats)
            AnsiConsole.MarkupLine($"[{Theme.Dim}]  {stats.Tokens} token{(stats.Tokens > 1 ? "s" : "")} · {stats.TokensPerSecond:0} tokens/s[/]");
        AnsiConsole.WriteLine();
    }

    private static void Write(char c) => AnsiConsole.Write(new Text(c.ToString(), Theme.Style(Theme.Text)));
}
