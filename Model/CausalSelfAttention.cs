using MiniChat.Config;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model
{

    /// <summary>
    /// Le cœur du Transformer : chaque token "regarde" les tokens précédents
    /// et décide lesquels sont importants pour prédire la suite.
    ///
    /// Pour chaque position on calcule 3 vecteurs :
    ///  - Query (Q) : "qu'est-ce que je cherche ?"
    ///  - Key   (K) : "qu'est-ce que je contiens ?"
    ///  - Value (V) : "qu'est-ce que je transmets si on me choisit ?"
    /// Score = Q · K. Plus le score est haut, plus on prend la Value de ce token.
    ///
    /// "Causal" = un token ne peut pas regarder le futur (sinon il tricherait).
    /// "Multi-têtes" = plusieurs attentions en parallèle, chacune apprend un rôle différent.
    /// Architecture Modern : Q et K sont tournés par RoPE (leur position), et la projection n'a pas de biais.
    ///
    /// Le calcul (scores, masque, softmax, moyenne des Values) est fait par
    /// functional.scaled_dot_product_attention : même résultat que l'écrire à la main,
    /// mais libtorch choisit un noyau optimisé, beaucoup plus rapide et économe en mémoire sur GPU.
    /// </summary>
    public sealed class CausalSelfAttention : Module<Tensor, Tensor>
    {
        private readonly TorchSharp.Modules.Linear _qkv;
        private readonly TorchSharp.Modules.Linear _projection;
        private readonly DropoutRate _dropout;   // le même pour tout le modèle (voir DropoutRate)
        private readonly int _heads;
        private readonly int _headDim;
        private readonly Rope? _rope;   // Modern seulement ; la même pour toutes les couches

        /// <param name="rope">Rotation des positions (Modern), null pour Classic.</param>
        public CausalSelfAttention(ModelConfig config, Rope? rope, DropoutRate dropout) : base(nameof(CausalSelfAttention))
        {
            if (config.EmbedDim % config.Heads != 0)
                throw new ArgumentException("EmbedDim doit être divisible par Heads.");

            _heads = config.Heads;
            _headDim = config.HeadDim;
            _rope = rope;

            // Une seule couche calcule Q, K et V d'un coup (plus rapide que trois couches).
            _qkv = Linear(config.EmbedDim, 3 * config.EmbedDim, hasBias: false);
            _projection = Linear(config.EmbedDim, config.EmbedDim, hasBias: rope is null);
            _dropout = dropout;

            RegisterComponents();
        }

        /// <summary>La projection de sortie : elle écrit dans la connexion résiduelle (voir MiniGpt, initialisation).</summary>
        public TorchSharp.Modules.Linear Output => _projection;

        /// <param name="x">Forme (batch, temps, EmbedDim).</param>
        public override Tensor forward(Tensor x) => Forward(x, cache: null, layer: 0);

        /// <param name="cache">
        /// Génération : les Keys/Values des tokens déjà vus (voir KvCache), x ne contient que les nouveaux tokens.
        /// null : x est toute la séquence (entraînement).
        /// </param>
        /// <param name="layer">Numéro de la couche : sa place dans le cache.</param>
        public Tensor Forward(Tensor x, KvCache? cache, int layer)
        {
            long batch = x.shape[0], time = x.shape[1], channels = x.shape[2];
            long start = cache?.Length ?? 0;   // position du premier token de x

            // 1. Calcul de Q, K, V puis découpage en têtes : (batch, têtes, temps, headDim)
            var parts = _qkv.forward(x).chunk(3, 2);
            var q = SplitHeads(parts[0], batch, time);
            var k = SplitHeads(parts[1], batch, time);
            var v = SplitHeads(parts[2], batch, time);

            if (_rope is not null)
            {
                q = _rope.Apply(q, start);
                k = _rope.Apply(k, start);
            }

            // Avec le cache : K et V de tous les tokens jusqu'ici, les anciens viennent du cache.
            // Le masque causal de scaled_dot_product_attention suppose que Q et K commencent au même token :
            // vrai sans cache ou cache vide. Sinon le nouveau token i voit les start + i + 1 premières positions.
            Tensor? mask = null;
            bool causal = true;
            if (cache is not null)
            {
                (k, v) = cache.Append(layer, k, v);
                if (start > 0)
                {
                    causal = false;
                    if (time > 1)
                        mask = arange(start + time, device: x.device).unsqueeze(0)
                            .le((arange(time, device: x.device) + start).unsqueeze(1));
                }
            }

            // 2 à 5 en un seul appel, équivalent à :
            //   scores  = Q · Kᵀ / sqrt(headDim)            (batch, têtes, temps, temps)
            //   scores  = masque causal (le futur à -∞)     un token ne voit pas la suite
            //   poids   = dropout(softmax(scores))          des pourcentages qui somment à 1
            //   sortie  = poids · V                         moyenne pondérée des Values
            // Arguments par position : dans TorchSharp, le paramètre "causal" s'appelle is_casual (faute de frappe).
            double dropout = training ? _dropout.Value : 0.0;   // pas de dropout en génération
            var attention = functional.scaled_dot_product_attention(q, k, v, mask, dropout, causal);

            // On recolle les têtes ensemble.
            var output = attention                        // (batch, têtes, temps, headDim)
                .transpose(1, 2).contiguous()             // (batch, temps, têtes, headDim)
                .view(batch, time, channels);             // (batch, temps, EmbedDim)

            return _dropout.Apply(_projection.forward(output), training);
        }

        private Tensor SplitHeads(Tensor t, long batch, long time) =>
            t.view(batch, time, _heads, _headDim).transpose(1, 2);
    }
}
