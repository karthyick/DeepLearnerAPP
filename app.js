const topics = [
    {
        main: "Neural Networks",
        subtopics: [
            {
                name: "Perceptron and ANN",
                details: {
                    studyMaterial: "https://www.tutorialspoint.com/artificial_neural_network/index.htm",
                    worldFamous: "https://www.youtube.com/watch?v=kft1AJ9WVDk",
                    indianFamous: "https://www.youtube.com/watch?v=2h0b-J1zmo4",
                    codeSamples: "https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py",
                    innovation: "https://deepmind.com/research/highlighted-research/neural-networks"
                }
            },
            {
                name: "Forward and Backward Propagation",
                details: {
                    studyMaterial: "https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html",
                    worldFamous: "https://www.youtube.com/watch?v=tIeHLnjs5U8",
                    indianFamous: "https://www.youtube.com/watch?v=XxCqkQ6irCU",
                    codeSamples: "https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/1_Introduction/fully_connected_feed.py",
                    innovation: "https://arxiv.org/abs/2002.06439"
                }
            },
            {
                name: "Activation Functions (ReLU, Sigmoid)",
                details: {
                    studyMaterial: "https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/",
                    worldFamous: "https://www.youtube.com/watch?v=-7scQpJT7uo",
                    indianFamous: "https://www.youtube.com/watch?v=Nk7iTjxg2fA",
                    codeSamples: "https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py",
                    innovation: "https://analyticsindiamag.com/activation-function-advancements-in-ai/"
                }
            },
            {
                name: "Loss Functions",
                details: {
                    studyMaterial: "https://deepnotes.io/softmax-crossentropy",
                    worldFamous: "https://www.youtube.com/watch?v=ErfnhcEV1O8",
                    indianFamous: "https://www.youtube.com/watch?v=saMxy7f3FfM",
                    codeSamples: "https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py",
                    innovation: "https://ai.googleblog.com/2021/01/fine-tuning-your-loss-functions.html"
                }
            },
            {
                name: "Optimization Algorithms (Gradient Descent, Adam Optimizer)",
                details: {
                    studyMaterial: "https://ruder.io/optimizing-gradient-descent/",
                    worldFamous: "https://www.youtube.com/watch?v=sDv4f4s2SB8",
                    indianFamous: "https://www.youtube.com/watch?v=lSro92NQNqE",
                    codeSamples: "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizers.py",
                    innovation: "https://blog.openai.com/adaptive-gradient-algorithms/"
                }
            }
        ]
    },
    {
        main: "Image Data Processing & Augmentation",
        subtopics: [
            {
                name: "Image Pre-processing",
                details: {
                    studyMaterial: "https://machinelearningmastery.com/image-pre-processing-for-deep-learning/",
                    worldFamous: "https://www.youtube.com/watch?v=xujm-t_7gKw",
                    indianFamous: "https://www.youtube.com/watch?v=xxd6kV20g88",
                    codeSamples: "https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py",
                    innovation: "https://arxiv.org/abs/1906.11364"
                }
            },
            {
                name: "Image Augmentation",
                details: {
                    studyMaterial: "https://towardsdatascience.com/image-augmentation-in-deep-learning-18cd44e0a7f5",
                    worldFamous: "https://www.youtube.com/watch?v=hKb3uONuZ2E",
                    indianFamous: "https://www.youtube.com/watch?v=ZGHXHts_TWw",
                    codeSamples: "https://github.com/aleju/imgaug",
                    innovation: "https://blog.roboflow.com/image-augmentation-in-machine-learning/"
                }
            }
        ]
    },
    {
        main: "Convolutional Neural Networks (CNN) & Computer Vision",
        subtopics: [
            {
                name: "Semantic Segmentation",
                details: {
                    studyMaterial: "https://www.coursera.org/learn/convolutional-neural-networks",
                    worldFamous: "https://www.youtube.com/watch?v=ZjM_XQa5s6s",
                    indianFamous: "https://www.youtube.com/watch?v=Ek8BUupcsQ",
                    codeSamples: "https://github.com/mrgloom/awesome-semantic-segmentation",
                    innovation: "https://arxiv.org/pdf/1505.04597.pdf"
                }
            }
        ]
    },
    {
        main: "Recurrent Neural Networks (RNN) & NLP",
        subtopics: [
            {
                name: "GRU",
                details: {
                    studyMaterial: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
                    worldFamous: "https://www.youtube.com/watch?v=WCUNPb-5EYI",
                    indianFamous: "https://www.youtube.com/watch?v=XdM6ER7zTLk",
                    codeSamples: "https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py",
                    innovation: "https://www.microsoft.com/en-us/research/blog/neural-language-representation-using-rnns/"
                }
            }
        ]
    },
    {
        main: "Generative Adversarial Networks (GANs)",
        subtopics: [
            {
                name: "GAN Architecture",
                details: {
                    studyMaterial: "https://keras.io/examples/generative/dcgan_overview/",
                    worldFamous: "https://www.youtube.com/watch?v=IHcEFIBZjiw",
                    indianFamous: "https://www.youtube.com/watch?v=9FSrXRjYgG0",
                    codeSamples: "https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py",
                    innovation: "https://arxiv.org/abs/1406.2661"
                }
            }
        ]
    },
    {
        main: "Transfer Learning",
        subtopics: [
            {
                name: "Dog vs Cat Classification using Transfer Learning Project",
                details: {
                    studyMaterial: "https://www.kaggle.com/c/dogs-vs-cats",
                    worldFamous: "https://www.youtube.com/watch?v=7GgfXd8kvgo",
                    indianFamous: "https://www.youtube.com/watch?v=cF9YYp_bJ2s",
                    codeSamples: "https://github.com/keras-team/keras/blob/master/examples/cifar10_transfer.py",
                    innovation: "https://arxiv.org/pdf/1801.02305.pdf"
                }
            }
        ]
    },
    {
        main: "Pre-Trained Models",
        subtopics: [
            {
                name: "GPT",
                details: {
                    studyMaterial: "https://arxiv.org/abs/2005.14165",
                    worldFamous: "https://www.youtube.com/watch?v=ILsA4nyG7I0",
                    indianFamous: "https://www.youtube.com/watch?v=gSTjkgid15k",
                    codeSamples: "https://github.com/openai/gpt-3",
                    innovation: "https://openai.com/research/gpt"
                }
            },
            {
                name: "LLM",
                details: {
                    studyMaterial: "https://arxiv.org/abs/2201.05163",
                    worldFamous: "https://www.youtube.com/watch?v=WTL9KZ_OasM",
                    indianFamous: "https://www.youtube.com/watch?v=eziE3i-JWBw",
                    codeSamples: "https://github.com/bigscience-workshop",
                    innovation: "https://huggingface.co/transformers/"
                }
            },
            {
                name: "RAN",
                details: {
                    studyMaterial: "https://arxiv.org/abs/2110.02705",
                    worldFamous: "https://www.youtube.com/watch?v=d8xW04c2yZY",
                    indianFamous: "https://www.youtube.com/watch?v=OhhnbvnmqZs",
                    codeSamples: "https://github.com/ran-model/repository",
                    innovation: "https://www.ran-ai.org"
                }
            }
        ]
    }
];

function createTile(subtopic) {
    const tile = document.createElement('div');
    tile.classList.add('tile');

    const title = document.createElement('h2');
    title.textContent = subtopic.name;
    tile.appendChild(title);

    const ul = document.createElement('ul');

    for (const [key, value] of Object.entries(subtopic.details)) {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${key}:</strong> ${value.includes('http') ? `<a href="${value}" target="_blank">Link</a>` : value}`;
        ul.appendChild(li);
    }

    tile.appendChild(ul);
    return tile;
}

function populateTiles() {
    const container = document.getElementById('tiles-container');

    topics.forEach(topic => {
        topic.subtopics.forEach(subtopic => {
            const tile = createTile(subtopic);
            container.appendChild(tile);
        });
    });
}

populateTiles();
