<h1 align="center">
  <img src="media/bee.png" alt="Logo" />
</h1>
<h2 align="center">
</h2>

<div align="center">

![img](https://img.shields.io/badge/Python-3.7-blue.svg?style=popout&logo=python)
![img](https://img.shields.io/badge/ElasticSearch-7.0.0-purple.svg?style=popout&logo=Elasticsearch)
![img](https://img.shields.io/badge/Kibana-7.7.0-purple.svg?style=popout&logo=Kibana)
![img](https://img.shields.io/badge/Jupyter-notebook-orange.svg?style=popout&logo=Jupyter)
![img](https://img.shields.io/badge/Docker-compose-blue.svg?style=popout&logo=docker)
</div>

### Overview


### Infra & Services
<div align="center">
<img  src="media/overview.png" alt="Logo"/>
</div>



### Resources
##### Python modules:
`joblib` , 
`matplotlib`,
`numpy`,
`pandas`,
`requests`,
`requests`,
`scikit`,
`seaborn`,
`nltk`,
`fasttext`,
`elasticsearch`,
`xmltodict`,

##### Other:
1. Notebook: https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html
2. Elastic: https://www.elastic.co/products/elastic-stack

### Containers:
| name of container         | needed for |
| ------------------      | ----------     |
| `elasticsearch`   | indexing vacancy data (`localhost:9200`)|
| `kibana` | Dashboard for metrics on the vacancy data (`localhost:5601`) |
| `scipy-notebook`  | notebook with scipy tools and access to the ML pipeline modules (`localhost:8888`) |

### Data Collection:
 [project Monarch](https://github.com/Yaleesa/project-Monarch)

### Requirements & Setup

1. Docker
2. `pip install requirements.txt`


