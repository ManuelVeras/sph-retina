# Organização Pessoal

## Inputs

###todo

corrigir esse warning: "/home/mstveras/mmdetection_2/sphdet/bbox/kent_formator.py:486: ComplexWarning: Casting complex values to real discards the imaginary part K[1:,1:] = eigvects"


entender como funciona bbox_targets (tem muitas linha srepeditas)

bbox_targets -> existem linhas (bonding kents) com valores de 0 em todas dims, pq isso acontece?

-> problemas na loss: entender limitacoes kappa e beta e se codigo comntempla isso e como resolver; debugar somente a loss nesse momento



-> ok, descobri que a o codigo do iou/loss da kent estava usando as coisas em ordem kappa, beta, phi, psi, eta, mas o codigo original do kent usa phi, psi, eta, kappa, beta....

-> pelo jeito o codigo do iou/loss esta recebendo em angulos, tambem e o certo seria radianos (será??)

loss retornando nan debugar essa póraaa
overlaps retornando nan na ultima dim


- valores negativos em get_kld - > old kent_loss esta concordando com kent_loss. debugar essa merda, alguns valores negativos sao proximos a zero, mas outros nao. os scripts concordam aparentemente. o problema é que ta dando vlaor negativo. coloquei ali um exmplo, acho que tem que fazer na mao pra entender oonde que ta o erro,

-sera que existe algum erro em usar a aproximação da constante e derivar???? - mesmo assim acho que podemos usar a série truncada , caso seja esse o problema


- pareque que quando kappa eh muito pequeno ou beta muito proximno de kappa/2, temos valores negativos

- ideias: limitar kappa a maior que x 

 - acho que python3 kent_acessories/get_kent_annotations.py est  sendo aplicada as kents e nao as bfovs




- setar threshold de bfov: tamanho minimo 8 graus na lat 





### Variáveis e Configurações
- **reg_decoded_bbox (bool)**: 
  - Se `True`, a regressão loss é aplicada diretamente em decoded bounding boxes, convertendo as caixas previstas e os alvos de regressão para o formato de coordenadas absolutas. 
  - DEFAULT: `False`. Deve ser `True` ao usar `IoULoss`, `GIoULoss` ou `DIoULoss` no bbox head.
  - **Nota**: Parece que essa variável deveria ser falsa.

### JSONs Importados
- **Formato**: Os JSONs importados pelo código (e.g., `instances_train.json`) estão em formato de `bfov`? E se estiverem no formato de bounding box?
- **NMS**: Precisa fazer algo? Inicialmente, pode ser omitido, mas como exatamente fazer isso?
- **bbox.sampler**: É relevante?

### Verificações
- **Arquivo**: Verificar se `bfov2kent_single_torch.py` está correto.
- **Linha 272**: seld.reg_decoded_box é usado na linha 272 de `anchor head.py`.
- **get_targets**: Onde é usado? Na loss. Onde a loss é usada?

### Código e Funções
- **Funções**: `bbox2delta` e `delta2bbox` precisam ser alteradas para fazer uma normalização que faça sentido para o caso das `kents`.
- **Classe**: Criar classe da loss, `iou` e `dataloader` no molde das classes de loss existentes.
- **Configuração**: Editar configuração para não computar `map`.

### Log de Atividades
- **Log (08/06)**:
  - Consegui rodar versão do código que converge com a branch (commit `b33fcbffbd980d09d7f494319f8d37523dde4c12`).
  - Problema com `pandora` nos commits seguintes (loss começava a dar `NaN` a partir de certo ponto).

### Papers e Leituras
- **Papers**:
  - "Can we trust bounding box annotations for object detection?"
  - "Towards Calibrated Hyper-Sphere Representation via Distribution Overlap Coefficient for Long-tailed Learning".
  - `Rotated Object Detection with Circular Gaussian Distribution`