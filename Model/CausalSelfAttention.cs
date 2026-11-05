using MiniChat.Config;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model
{

    /// <summary>
    /// Le cœur du Transformer : chaque caractère "regarde" les caractères précédents
    /// et décide lesquels sont importants pour prédire la suite.
    ///
    /// Pour chaque position on calcule 3 vecteurs :
    ///  - Query (Q) : "qu'est-ce que je cherche ?"
    ///  - Key   (K) : "qu'est-ce que je contiens ?"
    ///  - Value (V) : "qu'est-ce que je transmets si on me choisit ?"
    /// Score = Q · K. Plus le score est haut, plus on prend la Value de ce caractère.
    ///
    /// "Causal" = un caractère ne peut pas regarder le futur (sinon il tricherait).
    /// "Multi-têtes" = plusieurs attentions en parallèle, chacune apprend un rôle différent.
    /// </summary>
    public sealed class CausalSelfAttention : Module<Tensor, Tensor>
    {
        private readonly TorchSharp.Modules.Linear _qkv;
        private readonly TorchSharp.Modules.Linear _projection;
        private readonly TorchSharp.Modules.Dropout _attentionDropout;
        private readonly TorchSharp.Modules.Dropout _outputDropout;
        private readonly int _heads;
        private readonly int _headDim;

        public CausalSelfAttention(ModelConfig config) : base(nameof(CausalSelfAttention))
        {
            if (config.EmbedDim % config.Heads != 0)
                throw new ArgumentException("EmbedDim doit être divisible par Heads.");

            _heads = config.Heads;
            _headDim = config.EmbedDim / config.Heads;

            // Une seule couche calcule Q, K et V d'un coup (plus rapide que trois couches).
            _qkv = Linear(config.EmbedDim, 3 * config.EmbedDim, hasBias: false);
            _projection = Linear(config.EmbedDim, config.EmbedDim);
            _attentionDropout = Dropout(config.Dropout);
            _outputDropout = Dropout(config.Dropout);

            RegisterComponents();
        }

        /// <param name="x">Forme (batch, temps, EmbedDim).</param>
        public override Tensor forward(Tensor x)
        {
            long batch = x.shape[0], time = x.shape[1], channels = x.shape[2];

            // 1. Calcul de Q, K, V puis découpage en têtes : (batch, têtes, temps, headDim)
            var parts = _qkv.forward(x).chunk(3, 2);
            var q = SplitHeads(parts[0], batch, time);
            var k = SplitHeads(parts[1], batch, time);
            var v = SplitHeads(parts[2], batch, time);

            // 2. Scores d'attention : (batch, têtes, temps, temps)
            //    On divise par sqrt(headDim) pour garder des valeurs stables.
            var scores = q.matmul(k.transpose(-2, -1)) * (1.0 / Math.Sqrt(_headDim));

            // 3. Masque causal : on cache tout ce qui est au-dessus de la diagonale (le futur).
            var future = ones(time, time, device: x.device).tril().eq(0);
            scores = scores.masked_fill(future, float.NegativeInfinity);

            // 4. Softmax : les scores deviennent des pourcentages qui somment à 1.
            var weights = _attentionDropout.forward(functional.softmax(scores, -1));

            // 5. Moyenne pondérée des Values, puis on recolle les têtes ensemble.
            var output = weights.matmul(v)                // (batch, têtes, temps, headDim)
                .transpose(1, 2).contiguous()             // (batch, temps, têtes, headDim)
                .view(batch, time, channels);             // (batch, temps, EmbedDim)

            return _outputDropout.forward(_projection.forward(output));
        }

        private Tensor SplitHeads(Tensor t, long batch, long time) =>
            t.view(batch, time, _heads, _headDim).transpose(1, 2);
    }
}