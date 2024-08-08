        reg_decoded_bbox (bool): If true, the regression loss would be applied directly on decoded bounding boxes, converting both the predicted boxes and regression targets to absolute

coordinates format. Default False. It should be `True` when

using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.

me parece que essa merda deveria ser falsa

-> criar plano de backup dos oscips (git)

-> medir tempos com line prfiling







acrediot que esse seja o scipt vhave: sph_rcnn_head.py



ctz que os jsons importados pelo codigo (instances_train.json) tao em formato de bfov????? e se estiverem no formato de bounding box???

retinnet em principio nao usa roi  - > esta sendo usado o sph_rcnn_head????

nms precisa fazer algo? ahxco que no primeiro momento pode omitir





bbox.sampler é relevante????





encoder/decoder 

modificar json e overfitar a uma só imagem ou bem poucas

dockerfile e devcontainer





erro nos import preciso importar o bagulho def _sph_box2kent_transform(boxes, img_size):

    img_h, img_w = img_size

    from mmdetection_2.kent_utils.bfov2kent_single_torch import deg2kent

    return deg2kent(boxes, img_h, img_w)









checar se bfov2kent_single_torch.py esta certo ou nao, acho que sim, deveria estar, mas confirmar

usado na linha 272 de anchor head.py



Purpose: This code decides how to prepare the target bounding boxes for training.

Condition: It checks if self.reg_decoded_bbox is False.

If False, it means the bounding boxes need to be encoded (converted to a different format) before being used for training.

If True, it means the bounding boxes are already in the correct format and don't need encoding.

Action:

If encoding is needed, it uses self.bbox_coder.encode to convert the positive bounding boxes (pos_bboxes) and their corresponding ground truth boxes (pos_gt_bboxes).

If no encoding is needed, it directly uses the ground truth boxes.

]

versao do mmdet usada 2.28.2



https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/customize_models.html





wtf is sampling_result.pos_gt_bboxes???? (alternativa se  reg_decoded_box é falso)



onde a porra do get_targets é usado????? na loss!!! onde a porra da loss é usada???

-> se logar de ler a doc pq tudo isso faz parte do anchor_head.py que é um arquivo nativo do mmdet









pseudo plano de ação:

- é difcil de explorar as coisas indo pra frente no codigo de forma sequencial, mas de forma regressiva é mais facil! vai ser necessario editar as coisas de forma hierarquica, mas isso deve ser feito de forma regressiva! vou usar o git detalhando cada passo e commits extensivos feitos em docs de texto, mas a ideia é fazer de forma regressiva. agora descobri que a classe anchorhead tem um lugar mais alto na hierar quia em relacao ao encoder













nota:





> /home/mstveras/mmdetection-2.x/sphdet/bbox/coder/kent_coder.py(61)encode()

-> assert bboxes.size(-1) == gt_bboxes.size(-1) == 5

(Pdb) bboxes

tensor([[157.5000,   0.0000,  15.9099,  31.8198],

        [ 22.5000,   5.6250,  15.9099,  31.8198],

        [264.3750,   5.6250,  31.8198,  15.9099],

        [275.6250,   5.6250,  31.8198,  15.9099],

        [ 45.0000,  11.2500,  15.9099,  31.8198]])

(Pdb) gt_bboxes

tensor([[1.7459e+00, 2.1746e+00, 1.5786e+00, 1.5523e+02, 3.9790e-13],

        [8.0012e-01, 2.9534e+00, 1.5708e+00, 1.7625e+01, 1.5067e+00],

        [6.4632e-01, 7.4776e-01, 1.5708e+00, 2.6588e+01, 6.0676e+00],

        [1.1536e+00, 2.6753e+00, 1.5708e+00, 1.5979e+01, 6.0426e+00],

        [1.1536e+00, 2.6753e+00, 1.5708e+00, 1.5979e+01, 6.0426e+00]])



a porra das bboxes de regressao tem apenas 4 params, ou seja, a head precisa ser alterada antes do encodign/decoding, ja que o gt tem 5 params ( e de fato deve ter 5 params) 













Alterei random flip pra simplesmente nao fazer nada!





















ficar atento se 'SphResize, SPH random flip e normalizacao nao podem cagar desempenho no primeiro momento. sera que isso é necessario???? se sim como fazer?





Passos sugeridos pelo chat gpt



Data Loading: Validate the data loading process.

Preprocessing: Check preprocessing steps.

Bounding Box Conversion: Validate custom bounding box conversions.

Model Initialization: Ensure the model is initialized correctly.

Forward Pass: Validate the forward pass.

Loss Calculation: Ensure correct loss calculation.

Backward Pass and Optimization: Validate gradient computation and parameter updates.

Validation: Validate model performance on validation data.

Evaluation Metrics: Ensure evaluation metrics are correct







sph_anchor_generator.py <- box_formator.py(Planar2SphBoxTransform

) <-  _pix2sph_box_transform    ou _tan2sph_box_transform





Acredito que vai ser necessário adaptar sphdet/models/heads -> TODO: entender se é necessário, onde é usado e para que a função _get_bboxes_single

 

vai ser necessário criar classe da loss, iou e dataloader no molde das classes de loss que já existem

editar config (ou a parte do código que define isso) para não computar map

outras modificações???





_base_ = [

    '../_base_/custom_imports.py',

    '../_base_/default_runtime.py', #n precisa editar acho

    '../_base_/schedules/schedule_120e.py', #acho n precisa editar

    '../_base_/datasets/indoor360.py', #q porra é essa? parece que foi exlcuida do commit

    '../_base_/models/sph_retinanet_r50_fpn.py', #vai ter que editar

]















KentRetinaHead class without any code.

High-Level Overview

__init__: Initializes the class with a specific box version and other arguments.

2. _init_layers: Initializes the layers specific to this head.

3. _bbox_post_process: Post-processes bounding boxes, including non-maximum suppression (NMS).

_get_bboxes_single: Converts model outputs into bounding box predictions for a single image.

5. loss_single: Computes the loss for a single scale level.

Mid-Level Overview

__init__:

Initializes the KentRetinaHead with a box_version (either 4 or 5).

Calls the parent class (RetinaHead) initializer.

2. _init_layers:

Calls the parent class method to initialize common layers.

Initializes a convolutional layer for bounding box regression.

_bbox_post_process:

Concatenates multi-level scores, labels, and bounding boxes.

Optionally rescales bounding boxes.

Applies score factors if provided.

Performs NMS using either PlanarNMS or SphNMS based on configuration.

Returns the final bounding boxes and labels.

4. _get_bboxes_single:

Processes the outputs of a single image to generate bounding box predictions.

Applies score filtering and top-k selection.

Decodes bounding box predictions.

Calls _bbox_post_process to finalize the bounding boxes and labels.

5. loss_single:

Computes classification and regression losses for a single scale level.

Reshapes and permutes tensors as needed.

Applies the classification loss.

Decodes bounding boxes if necessary and applies the regression loss.

Returns the computed losses.















Geração de ancoras:



sphdet>bbox>anchor>sph_anchor_generator.py

Bastante simples até ond eeu entendi, a unica paerte chave é a chamada para Planar2SphBoxTransform (definida em sphdet.bbox>box_formator.py)vamos supor box_version =4 nesse primeito momento, dessa forma nao temos a achamada bfov2rbfov. O passo fundamental dessa função é chamada da função _pix2sph_box_transform (localizada no mesmo arquivo).

devemos definfir _pix2kent_transform







class SphBox2KentTransform:

    def __init__(self):

        self.transform = _sph_box2kent_transform

    def __call__(self, boxes, img_size=(512, 1024)):

        return self.transform(boxes, img_size)











bounding kent







funções bbox2delta e delta2bbox - > tem que ser alteradas pra fazer uma normalizacao que faça sentido para o nosso caso das kents!!!



















Summary of _bbox_post_process Function

The _bbox_post_process function is responsible for the final processing of bounding box predictions. Its main roles include:

Concatenation: Combines bounding boxes, scores, and labels from multiple levels into single tensors.

2. Rescaling: Optionally rescales the bounding boxes to the original image scale if rescale is True.

Score Adjustment: Adjusts scores using score factors if provided.

Non-Maximum Suppression (NMS): Applies NMS to remove redundant bounding boxes based on their Intersection over Union (IoU) scores.

Output: Returns the final bounding boxes and labels, either with or without NMS.

Potential Changes for Different Representation Formats

If you change the representation format from bounding boxes to something else (e.g., polygons, keypoints), the following changes might be necessary:

Concatenation:

Update the concatenation logic to handle the new format.

Example: Change mlvl_bboxes to mlvl_polygons if using polygons.

Rescaling:

Modify the rescaling logic to correctly adjust the new representation format.

Example: Update the rescaling formula to work with polygons.

NMS:

Adapt or replace the NMS method to work with the new representation.

Example: Use a different NMS algorithm suitable for polygons.

4. Decoding:

Ensure that any decoding logic before post-processing is updated to handle the new format.

Example: Update the bbox_coder.decode method if it is used elsewhere in the code.



























































































































acho que algumas boxes podem estar nao sendo convertidas, verifiquei que para algumas imagens algumas anotações parece que nao foram convertidas pra kent.



checar configs/_base_/models/sph_retinanet_r50_fpn.py ele tem a maior parte das peças chave













-  não poderíamos usar tudo em radiano?

- como é feito o matching no gausiann label distribution learning?













Ler tutoriais MMdet e documentar (como?)

loss

anchors

conversões

jsons no formato esperado

dataloader

visualizer

matching









Passos para a integração dos códigos:



Conversão de âncoras é preciso?*

Criar jsons com anotações no formato esperado OU fazer diretamente no código? qual a melhor abordagem? > acredito que o melhor seja de forma separada mesmo.

implementação de dataloader, tranformação coords, anchor generator







*aqui acho que entra outro ponto em discussão que é se é valido uma conversão dessas ancoras que nao simplesmente converter as ancoras do caso bfov pra kent. Seria bom entender como as ancoras sao definidas no caso esférico tambem….







====================================================================



Reunião cláudio e thiago dia 13/06





próximos passos



 - fazer conversão usando moment estimation



- limitar kappa entre 0 e x 



- ver como é feito estimação do ângulo feita pelos caras do mmdet, pq vamos usar coisas parecidas



adaptar kullback leibler pra usar na loss - começar pela que deu melhor para os caras da gaussian labreling





dicas/ hipoteses / apontamentos:



ao invés de regredir beta diretamente, talvez faça sentido de maneira indireta já que possui a limitação menor que metade do kappa…

Como regredir ângulos??

Como exatamente funciona o esquema dos caras do gaussian labeling? Eles regridem diretamente a bfov?

Qual efeito da arquitetura sobre o kent? Trabalho com Rômulo.

qual a  sensibilidade bfov <-> kent? bfovs parecidas geram kents parecidas tbm? 

depois de fazer o resto do pipeline: kent - > bfov, como faz? é importante ser um pra um, pode ser até empírico, pelo menos de inicio





======================================================================





















desvendar questao nao convergência do mmdet:



-rodar faster rcnn novamente e documentar melhor

- Terminar apresentação de andamento





mmdet loss do kld:

mudar loss para ponderada por latitude

adaptar get_kld para versao em pytorch 





Pesquisar por covariância estilo jeffri

Responder email do claudio



Usar só objetos maiores?











log jun 8:

consegui rodar versão do codigo que converge com a  com a branch ( commit b33fcbffbd980d09d7f494319f8d37523dde4c12 (HEAD)) 



agora lembrei que nao tava conseguindo rodar o codigo com o pandora, então deve ser esse o problema com os commits seguintes (loss começava a dar Nan a partir de certo ponto)



======================================================================





TO-DO: 26 - 29 maio





unificar códigos

Ler moment estimation: arquivos do thiago (appendix+kent+kasarapu). appendix mais claro







- bfov + kent overlay: ainda preciso unificar com tlts



- cálculo da kld, porem ainda nao sei como conferir 

	- atualização 24/05: conferi que o script retorna valores negativos de vez em quando e aparentemente troquei soma por subtração e vice versa em um momento

	- atualização 25/05: Fiz ajustes no cálculo do valor esperado (antes tava pegando vetor linha e o certo é vetor coluna). Revisei o código varias vezes. Asimetris estão satisfeitas. IDENTIDADE também. Tensor Flow tem método para calcular KL divergence entre duas VMf se setar b=0 no kent vira vmf então é um formula de validar. () -> Só serve para vmf e uniforme =(







KL divergence pra VMF: Towards Calibrated Hyper-Sphere Representation via Distribution Overlap Coefficient for Long-tailed Learning 





Na biblioteca (codigo) com utilities pra kent, phi e theta trocados e projeção diferente do que usamos, mas thiago mudou e já passou pra o codigo no dir “minimal”



- TODO: moment estimation: ler arquivos thiago, conferir script que ele fez e comparar com abordagem jeffri usando covariância



TODO:  olhar papers que citam kasarapu

=====================================================================



SSD



tentando treinar sem iniciar layer intermediárias; - ok

Resultados esquisitos - ok

validação funcionando - ok

SPH2POB USA 1024X512 -ok

MMdet - ok 

METAS SEMANA - traçar próximos passos -ok



REUNIÃO RECORRENTE





leitura oslo -OK







================================================================





reunião 27 março





consegui rodar treino com sph retina net _> funciona bem aparentemente



plots estão estranhos, não sei se existe codigo suplementar dele spara isso



dissertação - o que fazer?



vmf anisotrópico - n tem formula fechada



tratar areas undersampled como minority class -focal loss

BBs grandes acerta melhor?



to do no discord





TO DO SEMANA



RODAR MMDET TUTORIAL -ok 





dockerfile - funciona clonando repo; editar para copiar dir, mas  nao deve carregar dataset 



COMEÇAR A TRABALHAR MMDET SPH2POB

- devcontainer

- rodar retinanet como sanity check

- rodar código dos chineses (1. importar dataset)

rodar inferência, com imagens de exemplo - ok



LER OSLO 1 E 2 SECÇÃO

CHECAR SE EXISTE IOU PARA HEALPIX- 



mapa mental do mmdet códigos usados no sph2pob e instruções básicas



treinar rede com as 3 abordagens distintas de iou e gerar pequeno relatório para reunião com claudio/andamento com métricas disponíveis (mAP e recall)

	ok









certificar inferência no dataset de teste correto

custom loss: Existe sph-retinanet_ciou_loss será que funciona???

checar batch

ver implementações do sph2pob_iou eficientes, standard…

checar de novo questão losses. como é o ciou que funcionou?

Como outros papers fazem divisão de treino teste e val? foviou ou não usa validação, unbiased iou e sph2pob não é muito claro (acredito que nao use validação)



ver mm rot

batch =1 -> testei e nao deu certo



rodar treino com pandora: como modificar para receber 5 params?





======================================================================

 

Reunião 17 abril:





Dataloader foi retirado deles - acredito bastante que o problema esteja aqui mesmo

tive que criar sph_retinanet_r50_fpn_120e_pandora.py mas é um arquivo bastante simples, semelhante às outras configs porém usando como base sph_rotated (fornecido por eles)

browse dataSET SCRIPT

quanto grad norm é aceitável???

nao funcionou batch 1

Acredito que realmente seja problema no dataloader, pq loss de loc nao melhora (mas de classificação sim)

Mandar email pro chines

Rotated Object Detection with Circular Gaussian Distribution



=====================================================================



checar como é feito o cálculo do iou no teste -> * Verificar se a métrica usada para comparações é a "Unbiased". - ok mas seria bom revisar ainda

Checar nms

-similaridade,

-passar de uma representação para outra

loss



mandar email pro chines

- focar no paper!!!! (nao o preprint sla) jeffri (codigo tbm) e distribuição kent -> dúvidas pro claudio

- FASTERRCNN RODAR	

Can we trust bounding box annotations for object detection?

Vale a pena usar probiou aplicado as obbs usando sph2pob?







não usaram split pré definido.. entrar em contato com ele s(finger que queremos comparar) -  gaussian distribution labeling



Implementar Kent distribution:



1 - entender melhor fórmulas. Precisamos da constante normalização?

2 - gerar plot de amostras na esfera

3 - levar amostras para erp



conversões de/para BFoV e métrica para comparar duas distribuições.



























certificar que podemos usar kent

usar gbbs em erp já seria o suficiente?











IoU (Intersection over Union): This is the basic metric that just considers the area of overlap between the predicted bounding box and the ground truth (actual) bounding box.Generalized 



IoU (GIoU): This goes beyond just overlap and considers the area of the smallest enclosing box that fits both the predicted box and the ground truth box. It penalizes predictions that are far away from the ground truth.



Distance IoU (DIoU): This builds on GIoU by also considering the distance between the centers of the predicted box and the ground truth box. It encourages the predicted box to move closer to the center of the ground truth box.Complete 



IoU (CIoU): This is the most advanced metric, considering overlap, distance, and aspect ratio. It penalizes boxes that have a different aspect ratio from the ground truth box.



















Comandos usados:



1 - ACIONAMENTO DO DEV CONTAINER

IMPORTANTE



após acionar o dev container falta baixar as bibliotecas com:



pip install -v -e .

pip install yapf==0.40.1

pip install future tensorboard





2 - Experimentos com MMdet



Rodar treinamento



python3 tools/train.py configs/retinanet/sph_retinanet_r50_fpn_120e_indoor360.py 





Rodar teste e gerar relatório de mAP (usar flag recall para cálculo do recall)





python3 tools/test.py configs/retinanet/retinanet_r50_fpn_fp16_1x_coco.py checkpoints/best_bbox_mAP_50_epoch_85.pth  --eval mAP





Rodar teste e gerar arquivo pickle (usado no próximo passo)



python3 tools/test.py configs/retinanet/sph_retinanet_r50_fpn_120e_indoor360.py work_dirs/sph_retinanet_r50_fpn_120e_indoor360/best_bbox_mAP_50_epoch_85.pth --out results.pkl



Gerar imagens da predição



python3 tools/analysis_tools/analyze_results.pyconfigs/retinanet/sph_retinanet_r50_fpn_120e_indoor360.py results.pkl data/images





3 - shh com inf





ssh -oHostKeyAlgorithms=+ssh-dss mstveras@portal.inf.ufrgs.br -L8022:143.54.51.30:22



em outro terminal usar:



ssh mstveras@localhost -p8022







Conecte-se ao INF usando o VS Code e SSH

O SSH é um protocolo que permite conectar-se a outro computador remotamente. O VS Code possui uma extensão que permite usar o SSH para conectar-se a máquinas do INF.

Para se conectar ao INF usando o VS Code e SSH, siga estas etapas:

Instale a extensão Remote-SSH:


Abra o VS Code.

Clique em "Extensões" na barra lateral.

Pesquise por "Remote-SSH".

Clique em "Instalar".

Configure o SSH:


Crie um arquivo ~/.ssh/config (no Linux) ou %userprofile%\.ssh\config (no Windows).

Copie e cole o seguinte código no arquivo:

Host proxy

  HostName portal.inf.ufrgs.br

  HostKeyAlgorithms=+ssh-dss

  PubkeyAcceptedKeyTypes +ssh-rsa

  ForwardX11Trusted yes

  ForwardX11 yes

  Compression yes

  PreferredAuthentications publickey

  User <seu user no inf p/ acessar o proxy>

  ServerAliveInterval 240



Host <nome_maquina_no_inf>

  HostName <ip da maquina no inf>

  ForwardX11Trusted yes

  ForwardX11 yes

  Compression yes

  ProxyJump proxy

  User <teu user na maquina>

  ServerAliveInterval 240



* Substitua `<seu user no inf p/ acessar o proxy>` pelo seu nome de usuário do INF.

* Substitua `<nome_maquina_no_inf>` pelo nome da máquina que você deseja acessar.

* Substitua `<ip da maquina no inf>` pelo endereço IP da máquina que você deseja acessar.



Conecte-se à máquina:

Abra o VS Code.

Pressione Ctrl+Shift+P para abrir o Command Palette.

Digite "Connect to Host" e pressione Enter.

Digite o nome da máquina que você deseja acessar (por exemplo, <nome_maquina_no_inf>).

Na primeira tela, digite a senha do portal do INF.

Na segunda tela, digite a senha da máquina que você deseja acessar.











4 - Estrutura do mmdet



Certamente, a estrutura do arquivo de configuração do MMDet permite a personalização e por isso há uma relação com o arquivo '../base/models/sph_retinanet_r50_fpn.py'. algumas variáveis desse arquivo serão sobrescritas. 



Aqui está como funciona:

Estrutura da Configuração MMDet



O MMDet (e muitos sistemas de detecção similares) utilizam um sistema hierárquico de configuração para flexibilidade:



Arquivos Base:  Você possui uma coleção de arquivos de configuração base no diretório ../_base_. Eles estabelecem a estrutura central e padrões para vários componentes:



custom_imports.py: Gerencia qualquer módulo ou código personalizado necessário.

default_runtime.py: Configurações padrão de tempo de execução (por exemplo, registro, onde os checkpoints são salvos).

schedule_120e.py: Define o cronograma de treinamento (como a taxa de aprendizado muda ao longo do tempo).

indoor360.py: Especifica o dataset Indoor360 a ser usado.

sph_retinanet_r50_fpn.py: Define a arquitetura principal - um modelo RetinaNet com uma rede backbone ResNet-50 e Feature Pyramid Network (FPN).

Configuração Principal: Seu arquivo de configuração principal reúne esses arquivos base usando _base_ = [...]. Isso cria um ponto de partida.

Sobrescrevendo: A configuração principal pode então personalizar as coisas. Subscreve determinados valores dos arquivos base:

checkpoint_config e evaluation: Modifica a frequência com que os checkpoints são salvos e as avaliações são executadas.

log_config: Modifica a saída do log.

data: Ajusta o tamanho do batch e o processamento de dados.

model: Esta é a chave! Redefine partes da estrutura do modelo provenientes de sph_retinanet_r50_fpn.py:

Personaliza o gerador de anchors (box_formator)

Especifica funções de perda (FocalLoss, L1Loss)

Altera configurações para atribuir bounding boxes a anchors (a iou_calculator dentro do assigner)

Modifica como os resultados são avaliados durante o teste (iou_calculator, box_formator)

Em resumo

O arquivo sph_retinanet_r50_fpn.py configura a rede backbone do seu modelo.

Sua configuração principal importa isso, mantendo a estrutura geral, mas personalizando seletivamente partes-chave de como o modelo lida com bounding boxes e é treinado.

































checar s e problema nao é só no plot







analyze_results vs analyze_results_v2





browse dataset





usar hiperparams do paper









revisar e mandar email pro autor





















