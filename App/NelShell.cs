using MiniChat.Commands;
using MiniChat.UI;
using Spectre.Console;

namespace MiniChat.App;

/// <summary>
/// La boucle de Nel Code : le cadre d'accueil, puis chaque ligne tapée est soit une commande ("/train"),
/// soit un message pour le modèle. Une erreur est affichée, et on continue : la console ne se ferme pas.
/// </summary>
public sealed class NelShell
{
    private readonly CommandRegistry _registry = new();
    private readonly CommandContext _context;
    private readonly InputBox _input;

    public NelShell(Workbench workbench)
    {
        _context = new CommandContext(workbench, _registry);
        _input = new InputBox(_registry.Commands);
    }

    /// <param name="startupCommand">
    /// Commande à lancer tout de suite, venue de la ligne de commande (--train devient "/train"...). null = aucune.
    /// </param>
    public void Run(string? startupCommand = null)
    {
        WelcomeBanner.Show(_context.Workbench);

        if (startupCommand is not null)
        {
            ChatView.ShowUser(startupCommand);
            Execute(startupCommand);
        }

        while (!_context.QuitRequested)
        {
            string? input = _input.Read()?.Trim();
            if (input is null) break;
            if (input.Length == 0) continue;

            ChatView.ShowUser(input);
            Execute(input);
        }
    }

    private void Execute(string input)
    {
        // Échap pendant la commande ou la réponse : son token est annulé, elle s'arrête proprement.
        using var escape = EscapeKey.Watch();
        _context.Cancellation = escape.Token;
        try
        {
            if (input.StartsWith('/'))
            {
                _registry.Execute(_context, input);
                AnsiConsole.WriteLine();
            }
            else if (_context.Chat is { } chat)
            {
                ChatView.Reply(chat, input, escape.Token);
            }
            else
            {
                Messages.Warning("Pas encore de modèle pour te répondre : tape /train pour en entraîner un.");
                AnsiConsole.WriteLine();
            }
        }
        catch (UserException e)
        {
            Messages.Error(e.Message);
            AnsiConsole.WriteLine();
        }
        catch (Exception e)
        {
            // Plantage pendant un entraînement (mémoire GPU pleine...) : le checkpoint permet /resume.
            Messages.Error($"{e.GetType().Name} : {e.Message}");
            if (_context.Workbench.HasInterruptedTraining)
                Messages.Dim("L'entraînement est reprenable avec /resume.");
            AnsiConsole.WriteLine();
        }
        finally
        {
            _context.Cancellation = default;
        }
    }
}
