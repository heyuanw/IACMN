## __Iterated Attentive Convolution Matching Network (IACMN)__

This is an implementation of our CIKM 2019 paper: [Multi-Turn Response Selection in Retrieval-Based Chatbots with Iterated Attentive Convolution Matching Network].

## __Network__

IACMN is a neural deep matching network proposed for multi-turn response selection in the retrieval-based chatbot. 

IACMN iteratively constructs multi-grained representations of the response candidate and its multi-turn history context entirely based on hierarchical stacking of the proposed AGDR block, which is a refined combination of gated dilated-convolution and self-attention.

IACMN calculates and integrates the interactive matrices between each utterance-response pair from different views, then accumulates the sequencial matching vectors into a fused vector to obtain the final score.

- **Model Overview** 
<div align=center>
<img src="/appendix/model.png" width=800>
</div>

- **AGDR Block** 
<div align=center>
<img src="/appendix/AGDR_layer.jpeg" width=500>
</div>


## __Results__

We test IACMN on two large-scale multi-turn response selection tasks, i.e., the Ubuntu Corpus v1 and Douban Conversation Corpus, experimental results are bellow:

<img src="/appendix/result.png">

## __Usage__

First, please download data according to data/ReadMe.txt and unzip it:
```
cd data
unzip data.zip
```

Train and test the model by:
```
python main.py
```

## __Dependencies__

- Python >= 3.5
- Tensorflow >= 1.4

## __Citation__
If you use this code, please cite the following paper:

```
@inproceedings{wang2019multi,
  title={Multi-Turn Response Selection in Retrieval-Based Chatbots with Iterated Attentive Convolution Matching Network},
  author={Wang, Heyuan and Wu, Ziyi and Chen, Junyu},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={1081--1090},
  year={2019},
  organization={ACM}
}
```
