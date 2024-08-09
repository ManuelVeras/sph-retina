# Organização Pessoal

## Inputs

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
