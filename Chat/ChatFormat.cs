using System.Text;
using static MiniChat.Tokenization.BpeTokenizer;

namespace MiniChat.Chat;

/// <summary>
/// Le format des conversations que voit le modèle, avec les tokens spéciaux du tokenizer :
///   &lt;|doc|&gt;&lt;|user|&gt;Bonjour !&lt;|end|&gt;&lt;|assistant|&gt;Bonjour, comment vas-tu ?&lt;|end|&gt;&lt;|user|&gt;...
/// &lt;|doc|&gt; ouvre chaque conversation (comme chaque document de texte.txt). Un message tient entre son token
/// de début et &lt;|end|&gt; : il peut contenir des retours à la ligne, et c'est &lt;|end|&gt; qui arrête la réponse.
///
/// Le même format partout, sinon le modèle ne reconnaîtrait pas en discutant ce qu'il a appris :
///   - le fine-tuning (FromDialogueFile, ReplyWeights),
///   - le chat (ChatSession construit son prompt avec User, Assistant et Turn).
///
/// Sur le disque, dialogues.txt reste lisible et modifiable à la main (Toi: / IA:, une ligne vide
/// entre deux conversations) : il n'est converti qu'au moment du fine-tuning.
/// Les noms Toi et IA restent aussi ceux de l'affichage (ChatSession.UserPrefix, BotPrefix).
/// </summary>
public static class ChatFormat
{
    public static string User(string text) => UserToken + text.Trim() + EndToken;

    public static string Assistant(string text) => AssistantToken + text.Trim() + EndToken;

    /// <summary>Un tour complet : la question et sa réponse.</summary>
    public static string Turn(string user, string reply) => User(user) + Assistant(reply);

    /// <summary>
    /// dialogues.txt (lignes "Toi: ..." et "IA: ...") au format du modèle.
    /// Une ligne qui ne commence ni par "Toi:" ni par "IA:" continue le message d'avant (retour à la ligne gardé).
    /// Une ligne vide termine la conversation : la suivante commence par &lt;|doc|&gt;.
    /// </summary>
    public static string FromDialogueFile(string text)
    {
        var output = new StringBuilder();
        var message = new StringBuilder();
        string? role = null;           // UserToken, AssistantToken, ou null : pas de message en cours
        bool conversationOpen = false;

        void EndMessage()
        {
            if (role is not null)
                output.Append(role).Append(message.ToString().Trim()).Append(EndToken);
            role = null;
            message.Clear();
        }

        foreach (string line in text.Split('\n'))
        {
            string? newRole = line.StartsWith(ChatSession.UserPrefix.TrimEnd()) ? UserToken
                : line.StartsWith(ChatSession.BotPrefix.TrimEnd()) ? AssistantToken
                : null;

            if (newRole is not null)
            {
                EndMessage();
                if (!conversationOpen)
                {
                    output.Append(DocToken);
                    conversationOpen = true;
                }
                role = newRole;
                message.Append(line[(newRole == UserToken ? ChatSession.UserPrefix : ChatSession.BotPrefix).TrimEnd().Length..]);
            }
            else if (line.Trim().Length == 0)
            {
                EndMessage();
                conversationOpen = false;
            }
            else if (role is not null)
            {
                message.Append('\n').Append(line);
            }
        }
        EndMessage();
        return output.ToString();
    }

    /// <summary>
    /// 1 pour chaque caractère d'une réponse de l'IA et de son &lt;|end|&gt;, 0 ailleurs :
    ///   &lt;|user|&gt;Salut&lt;|end|&gt;&lt;|assistant|&gt;Coucou&lt;|end|&gt;
    ///   0000000000000000000000000000000001111111111111
    /// Le modèle lit tout, mais n'est noté que sur ce qu'il devra écrire : la réponse, puis &lt;|end|&gt;
    /// pour apprendre à s'arrêter. Les questions, c'est l'humain qui les tape.
    /// </summary>
    public static float[] ReplyWeights(string formatted)
    {
        var weights = new float[formatted.Length];
        int start = formatted.IndexOf(AssistantToken, StringComparison.Ordinal);
        while (start >= 0)
        {
            int replyStart = start + AssistantToken.Length;
            int end = formatted.IndexOf(EndToken, replyStart, StringComparison.Ordinal);
            int replyEnd = end < 0 ? formatted.Length : end + EndToken.Length;   // réponse coupée : jusqu'au bout
            for (int i = replyStart; i < replyEnd; i++)
                weights[i] = 1;
            start = end < 0 ? -1 : formatted.IndexOf(AssistantToken, replyEnd, StringComparison.Ordinal);
        }
        return weights;
    }
}
