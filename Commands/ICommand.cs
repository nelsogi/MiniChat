namespace MiniChat.Commands;

/// <summary>
/// Une commande de Nel Code (/help, /train...). Chaque classe qui implémente ICommand est trouvée
/// toute seule par CommandRegistry : pour ajouter une commande, il suffit d'ajouter sa classe.
/// </summary>
public interface ICommand
{
    /// <summary>Le nom tapé après "/", ex. "train".</summary>
    string Name { get; }

    /// <summary>Une ligne pour /help.</summary>
    string Description { get; }

    /// <summary>Comment l'écrire, ex. "/continue &lt;étapes&gt; [finetune] [--lr 3e-4]".</summary>
    string Usage { get; }

    /// <param name="args">Les mots tapés après le nom de la commande.</param>
    void Execute(CommandContext context, string[] args);
}
