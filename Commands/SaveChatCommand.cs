using System.Text;
using MiniChat.App;
using MiniChat.UI;

namespace MiniChat.Commands;

/// <summary>
/// Sauvegarde la conversation dans un fichier texte :
///   /save-chat                 chats/&lt;date&gt;.txt, dans le dossier des modèles
///   /save-chat bonjour         chats/bonjour.txt
///   /save-chat D:\chat.txt     ce fichier-là
/// </summary>
public sealed class SaveChatCommand : ICommand
{
    public string Name => "save-chat";
    public string Description => "Sauvegarde la conversation en texte (dans chats/, ou le fichier donné)";
    public string Usage => "/save-chat [fichier]";

    public void Execute(CommandContext context, string[] args)
    {
        var chat = context.RequireChat();
        if (chat.TurnCount == 0)
            throw new UserException("La conversation est vide : rien à sauvegarder.");

        string name = args.Length > 0 ? string.Join(' ', args) : DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        if (!Path.HasExtension(name))
            name += ".txt";
        // Un nom seul (ou relatif) va dans chats/ ; un chemin complet est pris tel quel.
        string path = Path.GetFullPath(Path.Combine(context.Workbench.ModelsFolder, "chats", name));
        bool replaced = File.Exists(path);

        var model = context.Workbench.Model!;
        var text = new StringBuilder();
        text.AppendLine($"Nel Code - conversation du {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        text.AppendLine($"Modèle : '{context.Workbench.ModelSource}' | {model.ParameterCount:N0} paramètres");
        text.AppendLine($"Réglages : température {chat.Temperature}, top-k {chat.TopK}, top-p {chat.TopP}");
        foreach (string turn in chat.Turns)
        {
            text.AppendLine();
            text.Append(turn);
        }

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text.ToString(), new UTF8Encoding(false));
        Messages.Success($"Conversation ({chat.TurnCount} tour(s)) {(replaced ? "sauvegardée, ancien fichier remplacé" : "sauvegardée")} : {path}");
    }
}
