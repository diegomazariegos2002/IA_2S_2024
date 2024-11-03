// Generates the main logic for the application
// List of algorithms to be used in the application
const algorithms = [
    { value: 'linear_regression', text: 'Linear Regression' },
    { value: 'decision_tree', text: 'Decision Tree' },
    { value: 'kmeans', text: 'K-Means' },
    { value: 'polynomial_regression', text: 'Polynomial Regression' },
    { value: 'naive_bayes', text: 'Naive Bayes' },
    { value: 'neuronal_network', text: 'Neural Network' },
    { value: 'knn', text: 'K-Nearest Neighbors' }
];

// Reference to the list box of algorithms
const selectElement = document.getElementById('algoritmoSeleccionado');

// Add the algorithms to the list box dinamically
algorithms.forEach(algorithm => {
    const option = document.createElement('option');
    option.value = algorithm.value;
    option.textContent = algorithm.text;
    selectElement.appendChild(option);
});

// Variables globales 

// Variables para la regresión
let xTrain, yTrain, xTest, yTest, predictionsTrain, predictionsTest;

// Variables para el árbol de decisión
let decisionTreeModel, dotStr, root, dTPredictions;

// Variables para KMeans
let kTemp_ms, iter_ms, data_ms, kMeans_ms, clusterizedData_ms;

let headers = [];
let chart;
let data1_1 = [];
let data2_0 = [];
let data1_0 = [];
let data4_0 = [];

// Atrributes or elements from the HTML
const algothSelect = document.getElementById('algoritmoSeleccionado');
const dtStInput1 = document.getElementById('dtStInput1');
const dtStInput2 = document.getElementById('dtStInput2');
const dtStInput3 = document.getElementById('dtStInput3');
const axleX = document.getElementById('axleX');
const axleY = document.getElementById('axleY');
const polyDegreeI = document.getElementById('polyDegree');
const trainPertgI = document.getElementById('trainPertg');
const trainBn = document.getElementById('trainBn');
const predictBn = document.getElementById('predictBn');
const graphBn = document.getElementById('graphBn');
const graphBn2 = document.getElementById('graphBn2');
const patternBn = document.getElementById('patternBn');
const predictBn2 = document.getElementById('predictBn2');
const file1 = document.getElementById('file1');
const file2 = document.getElementById('file2');
const file3 = document.getElementById('file3');
const clSelect = document.getElementById('clmnSelect');
const polyDgrContainer = document.getElementById('polyDgrContainer');
const trainPrtCntainer = document.getElementById('trainPrtCntainer');
const nvBayesFrmCntainer = document.getElementById('nvBayesFrmCntainer');
const resCntainer = document.getElementById('resCntainer');
const nvBayesPrdctionCntainer = document.getElementById('nvBayesPrdctionCntainer');
const neuralNkCntainer = document.getElementById('neuralNkCntainer');
const chrtCntainer = document.getElementById('chrtCntainer');
const treeCntainer = document.getElementById('treeCntainer');
const treeDiv = document.getElementById('tree');
const resPredCntainer = document.getElementById('resPredictionCntainer');
const resPred = document.getElementById('predictionText');

// Initialize events
function init() {
    algothSelect.addEventListener('change', updateVisibility);
    dtStInput1.addEventListener('change', handleFile1);
    dtStInput2.addEventListener('change', handleFile2);
    dtStInput3.addEventListener('change', handleFile3);
    trainBn.addEventListener('click', handleTrains);
    predictBn.addEventListener('click', handlePredicts);
    graphBn.addEventListener('click', handleShowCharts);
    patternBn.addEventListener('click', handleCalcPatterns);
    predictBn2.addEventListener('click', handlePredicts);
    graphBn2.addEventListener('click', handleShowCharts);

    updateVisibility();
}

function updateVisibility() {
    console.log('update: ', algothSelect.value);
    const selectedAlgorithm = algothSelect.value;

    file1.style.display = ['decision_tree', 'linear_regression', 'polynomial_regression', 'naive_bayes', 'neuronal_network', 'kmeans', 'knn'].includes(selectedAlgorithm) ? 'block' : 'none';
    file2.style.display = ['decision_tree', 'neuronal_network', 'kmeans', 'knn'].includes(selectedAlgorithm) ? 'block' : 'none';
    file3.style.display = selectedAlgorithm === 'neuronal_network' ? 'block' : 'none';
    clSelect.style.display = ['linear_regression', 'polynomial_regression'].includes(selectedAlgorithm) ? 'block' : 'none';
    polyDgrContainer.style.display = selectedAlgorithm === 'polynomial_regression' ? 'block' : 'none';
    trainPrtCntainer.style.display = ['linear_regression', 'polynomial_regression'].includes(selectedAlgorithm) ? 'block' : 'none';
    nvBayesFrmCntainer.style.display = selectedAlgorithm === 'naive_bayes' ? 'block' : 'none';
    resCntainer.style.display = 'none';
    nvBayesPrdctionCntainer.style.display = 'none';
    neuralNkCntainer.style.display = 'none';
    chrtCntainer.style.display = 'none';
    treeCntainer.style.display = 'none';
    predictBn.style.display = 'none';
    graphBn.style.display = 'none';
    graphBn2.style.display = 'none';
    patternBn.style.display = 'none';
    resPredCntainer.style.display = 'none';
    resPred.style.display = 'none';
    predictBn2.style.display = 'none';
    treeDiv.innerHTML = '';
}

function handleFile1(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            processCSV(text, 1);
        };
        reader.readAsText(file);
    }
}

function handleFile2(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            processCSV(text, 2);
        };
        reader.readAsText(file);
    }
}

function handleFile3(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            processCSV(text, 3);
        };
        reader.readAsText(file);
    }
}

function processCSV(csv, csvNumber) {
    const rows = csv.split("\n").map(row => row.split(",").map(cell => cell.trim()));
    const selectedAlgorithm = algothSelect.value;

    if (rows.length > 0) {
        if (csvNumber === 1) {
            headers = rows[0];
            data1_1 = rows.slice(1);
            data1_0 = rows.slice(0);
            if (selectedAlgorithm === "naive_bayes") {
                generateNaiveBayesForm(headers);
            }
            updateColumnSelectors(headers);
        } else if (csvNumber === 2) {
            data2_0 = rows.slice(0);
        } else if (csvNumber === 3) {
            data4_0 = rows.slice(0);
        }
    }
}

function updateColumnSelectors(headers) {
    axleX.innerHTML = '';
    axleY.innerHTML = '';

    headers.forEach((header, index) => {
        const option = document.createElement("option");
        option.value = index;
        option.textContent = header;
        axleX.appendChild(option);

        const optionY = document.createElement("option");
        optionY.value = index;
        optionY.textContent = header;
        axleY.appendChild(optionY);
    });
}

function generateNaiveBayesForm(headers) {
    nvBayesFrmCntainer.innerHTML = '';
    const attributesLabel = document.createElement('label');
    attributesLabel.textContent = 'Enter values for attributes:';
    nvBayesFrmCntainer.appendChild(attributesLabel);
    nvBayesFrmCntainer.appendChild(document.createElement('br'));

    headers.slice(0, -1).forEach((header, index) => {
        const input = document.createElement('input');
        input.type = 'text';
        input.id = `attribute_${index}`;
        input.name = 'attributes';
        input.placeholder = `Value for ${header}`;

        const label = document.createElement('label');
        label.textContent = `${header}: `;
        label.htmlFor = `attribute_${index}`;

        nvBayesFrmCntainer.appendChild(label);
        nvBayesFrmCntainer.appendChild(input);
        nvBayesFrmCntainer.appendChild(document.createElement('br'));
    });
}

function handleTrains() {
    const selectedAlgorithm = algothSelect.value;
    const xIndex = parseInt(axleX.value);
    const yIndex = parseInt(axleY.value);
    const trainPerc = parseFloat(trainPertgI.value) / 100;

    if (!((data1_0.length > 0) && (xIndex != null || yIndex != null) && (trainPerc > 0 && trainPerc <= 1))) {
        alert("Please load a valid CSV file and select columns if applicable.");
        return;
    }

    switch (selectedAlgorithm) {
        case 'linear_regression':
        case 'polynomial_regression':
            handleTrainRegreAlgos(selectedAlgorithm, xIndex, yIndex, trainPerc);
            break;
        case 'decision_tree':
            handleTrainDecisionTree();
            break;
        case 'naive_bayes':
            handleNaiveBayes();
            break;
        case 'neuronal_network':
            handleNeuralNetwork();
            break;
        case 'kmeans':
            handleTrainKMeans();
            break;
        case 'knn':
            handleKNN();
            break;
    }
}

function handlePredicts() {
    const selectedAlgorithm = algothSelect.value;
    switch (selectedAlgorithm) {
        case 'linear_regression':
        case 'polynomial_regression':
            handlePredicRegressions(selectedAlgorithm);
            break;
        case 'decision_tree':
            handlePredDecisionTree();
            break;
        case 'naive_bayes':
            break;
        case 'neuronal_network':
            break;
        case 'kmeans':
            break;
        case 'knn':
            break;
        default:
            console.log('Invalid algorithm');
            alert('Invalid algorithm');
            break;
    }
}

function handleShowCharts() {
    const selectedAlgorithm = algothSelect.value;
    switch (selectedAlgorithm) {
        case 'linear_regression':
        case 'polynomial_regression':
            showRegressionChart(selectedAlgorithm);
            break;
        case 'decision_tree':
            showDecisionTreeGraph();
            break;
        case 'naive_bayes':
            break;
        case 'neuronal_network':
            break;
        case 'kmeans':
            data_ms = data2_0.map(line => line.map(Number));
            if (data_ms[0].length === 1) {
                showKMeansChart1D();
            } else if (data_ms[0].length === 2) {
                showKMeansChart2D();
            } else {
                alert("Kmeans solo funciona con datos de 1 o 2 dimensiones.");
            }
            break;
        case 'knn':
            break;
        default:
            console.log('Invalid algorithm');
            alert('Invalid algorithm');
            break;
    }
}

function handleCalcPatterns() {
    const selectedAlgorithm = algothSelect.value;

    if (data2_0.length < 1) {
        alert("Please load a valid CSV file and select columns if applicable.");
        return;
    }

    switch (selectedAlgorithm) {
        case 'linear_regression':
            break;
        case 'polynomial_regression':
            break;
        case 'decision_tree':
            handlePatternDecisionTree();
            break;
        case 'naive_bayes':
            break;
        case 'neuronal_network':
            break;
        case 'kmeans':
            handleCalcPatternsKMeans();
            break;
        case 'knn':
            break;
        default:
            console.log('Invalid algorithm');
            alert('Invalid algorithm');
            break;
    }
}

// Función para manejar el entrenamiento
function handleTrainRegreAlgos(algorithm, xIndex, yIndex, trainPercentage) {
    console.log({ 'Algorithm': algorithm, 'XIndex': xIndex, 'YIndex': yIndex, 'TrainPercentage': trainPercentage });

    const xValues = data1_1.map(row => parseFloat(row[xIndex]));
    const yValues = data1_1.map(row => parseFloat(row[yIndex]));

    // Define el tamaño del conjunto de entrenamiento
    const trainSize = Math.floor(xValues.length * trainPercentage);

    // Divide los datos en conjuntos de entrenamiento y prueba
    xTrain = xValues.slice(0, trainSize);
    yTrain = yValues.slice(0, trainSize);
    xTest = xValues.slice(trainSize);
    yTest = yValues.slice(trainSize);

    resCntainer.style.display = 'block';

    console.log({ 'TrainSize': trainSize, 'xTrain': xTrain, 'yTrain': yTrain, 'xTest': xTest, 'yTest': yTest });

    // Habilitar el botón de predecir
    predictBn.style.display = 'block';
}

// Función para manejar la predicción
function handlePredicRegressions(algorithm) {
    if (algorithm === 'linear_regression') {
        const linearModel = new LinearRegression();
        linearModel.fit(xTrain, yTrain);
        predictionsTrain = linearModel.predict(xTrain); // Predicciones en los datos de entrenamiento
        predictionsTest = xTest.length > 0 ? linearModel.predict(xTest) : []; // Predicciones en los datos de prueba
    } else {
        const degree = parseInt(polyDegreeI.value);
        const polynomialModel = new PolynomialRegression();
        polynomialModel.fit(xTrain, yTrain, degree);
        predictionsTrain = polynomialModel.predict(xTrain);
        predictionsTest = xTest.length > 0 ? polynomialModel.predict(xTest) : [];
    }

    // Habilitar el botón de graficar
    graphBn.style.display = 'block';
}

// Función para manejar la visualización de la gráfica
function showRegressionChart(algorithm) {
    const ctx = document.getElementById('chart1').getContext('2d');
    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Training Data',
                    data: xTrain.map((x, i) => ({ x: x, y: yTrain[i] })),
                    backgroundColor: 'rgba(255, 99, 132, 0.8)'
                },
                {
                    label: 'Test Data',
                    data: xTest.map((x, i) => ({ x: x, y: yTest[i] })),
                    backgroundColor: 'rgba(54, 162, 235, 0.8)'
                },
                {
                    label: 'Predictions on Training Data',
                    data: xTrain.map((x, i) => ({ x: x, y: predictionsTrain[i] })),
                    type: 'line',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    fill: false
                },
                {
                    label: 'Predictions on Test Data',
                    data: xTest.map((x, i) => ({ x: x, y: predictionsTest[i] })),
                    type: 'line',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: algorithm === 'linear_regression' ? 'Linear Regression' : 'Polynomial Regression'
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            }
        }
    });

    chrtCntainer.style.display = 'block';
    treeCntainer.style.display = 'none';
}

function handleTrainDecisionTree() {
    console.log({ 'Data3': data1_0, 'Data2': data2_0 });

    decisionTreeModel = new DecisionTreeID3(data1_0);
    root = decisionTreeModel.train(decisionTreeModel.dataset);

    resCntainer.style.display = 'block';

    console.log('results', { 'Decision Tree': root });

    graphBn.style.display = 'block';
}

function showDecisionTreeGraph() {
    console.log('showDecisionTreeGraph');

    dotStr = decisionTreeModel.generateDotString(root);
    treeDiv.innerHTML = '';

    const parsedData = vis.network.convertDot(dotStr);
    const data = {
        nodes: parsedData.nodes,
        edges: parsedData.edges
    };
    const options = {
        layout: {
            hierarchical: {
                direction: 'UD',
                sortMethod: 'directed'
            }
        },
        edges: {
            smooth: true
        },
        physics: {
            enabled: false
        }
    };

    new vis.Network(treeDiv, data, options);
    chrtCntainer.style.display = 'none';
    treeCntainer.style.display = 'block';
    patternBn.style.display = 'block';
}

function handlePatternDecisionTree() {
    console.log('Entrando a btn Patterns ', { 'Data3': data1_0, 'Data2': data2_0 });

    dTPredictions =
        (data2_0 != null || (data2_0.length > 0 && data2_0.length < 2)) ? decisionTreeModel.predict([data2_0[0], data2_0[1]], root) : null;


    alert('Patrones calculados con éxito.');

    predictBn2.style.display = 'block';
}

function handlePredDecisionTree() {
    console.log(
        'Entrando a btn Predicts ',
        { 'Data3': data1_0, 'Data2': data2_0, 'Root': root, 'Predictions': dTPredictions });

    if (dTPredictions && dTPredictions.value) {
        resPred.value = `${data1_0[0][data1_0[0].length-1]} : ${dTPredictions.value}`;
    } else {
        resPred.value = 'No se pudo obtener la predicción.';
    }

    resPredCntainer.style.display = 'block';
    resPred.style.display = 'block';
}


function handleNaiveBayes() {
    let model = new BayesMethod();
    const attributes = data1_1.map(row => row.slice(0, -1));
    const classes = data1_1.map(row => row.slice(-1)[0]);

    headers.slice(0, -1).forEach((header, index) => {
        let columnData = attributes.map(row => row[index]);
        model.addAttribute(columnData, header);
    });

    model.addClass(classes, headers[headers.length - 1]);
    model.train();

    let inputValues = [];
    headers.slice(0, -1).forEach((header, index) => {
        let inputValue = document.getElementById(`attribute_${index}`).value;
        inputValues.push(inputValue);
    });

    const prediction = model.predict(inputValues);
    showNaiveBayesPrediction(prediction);
}

function handleTrainKMeans() {
    console.log('handleTrainKMeans', { 'Data1_1': data1_1, 'Data2': data2_0, 'Data1_0': data1_0 });
    
    [kTemp_ms, iter_ms] = data1_0[0].map(Number);
    data_ms = data2_0.map(line => line.map(Number));

    if (data_ms[0].length === 1) {
        kMeans_ms = new LinearKMeans(kTemp_ms, data_ms.flat());
        resCntainer.style.display = 'block';
        patternBn.style.display = 'block';
    } else if (data_ms[0].length === 2) {
        kMeans_ms = new _2DKMeans(kTemp_ms, data_ms);
        resCntainer.style.display = 'block';
        patternBn.style.display = 'block';
    } else {
        alert("Kmeans solo funciona con datos de 1 o 2 dimensiones.");
    }
}

function handleCalcPatternsKMeans() {
    console.log('handleCalcPatternsKMeans', { 'Data1_1': data1_1, 'Data2': data2_0, 'Data1_0': data1_0 });
    if (data_ms[0].length === 1) {
        clusterizedData_ms = kMeans_ms.clusterize(kTemp_ms, data_ms.flat(), iter_ms);
        graphBn2.style.display = 'block';
        alert("Patrones calculados con éxito.");
    } else if (data_ms[0].length === 2) {
        clusterizedData_ms = kMeans_ms.clusterize(kTemp_ms, data_ms, iter_ms);
        graphBn2.style.display = 'block';
        alert("Patrones calculados con éxito.");
    } else {
        alert("Kmeans solo funciona con datos de 1 o 2 dimensiones.");
        return;
    }
}

function showKMeansChart1D() {
    console.log('showLinearKMeansChart', { 'Clusterized Data': clusterizedData_ms, 'K': kTemp_ms });
    
    const chartDom = document.getElementById('chart1').getContext('2d');
    if (chart) {
        chart.destroy();
    }

    const clustersIter = new Set(clusterizedData_ms.map(a => a[1]));
    console.log('Clusters', clustersIter);
    
    const colorsPoints = Array.from(clustersIter).map(() =>
        '#' + Math.floor(Math.random() * 16777215).toString(16)
    );
    console.log('Colors', colorsPoints);

    const dataSetsGraph = Array.from(clustersIter).map((cluster, index) => ({
        label: `Cluster ${cluster}`,
        data: clusterizedData_ms.filter(d => d[1] === cluster).map(d => ({ x: d[0], y: 0 })),
        backgroundColor: colorsPoints[index],
        pointRadius: 5
    }));

    dataSetsGraph.push({
        label: 'Centroids',
        data: Array.from(clustersIter).map(value => ({ x: value, y: 0 })),
        backgroundColor: '#ff0000',
        pointRadius: 10
    });

    console.log('Datasets', dataSetsGraph);

    chart = new Chart(chartDom, {
        type: 'scatter',
        data: {
            datasets: dataSetsGraph
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: 'KMeans (1D)'
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                },
                y: {
                    display: false
                }
            }
        }
    });
    chrtCntainer.style.display = 'block';
    treeCntainer.style.display = 'none';
}

function showKMeansChart2D() {
    console.log('showKMeansChart2D', { 'Clusterized Data': clusterizedData_ms, 'K': kTemp_ms });

    const chartDom = document.getElementById('chart1').getContext('2d');
    if (chart) {
        chart.destroy();
    }

    const clusters = new Set(clusterizedData_ms.map(a => a[1]));
    console.log('Clusters', clusters);

    const colors = Array.from(clusters).map(() =>
        '#' + Math.floor(Math.random() * 16777215).toString(16)
    );
    console.log('Colors', colors);

    const dataSetsGraph = Array.from(clusters).map((cluster, index) => ({
        label: `Cluster ${cluster}`,
        data: clusterizedData_ms.filter(d => d[1] === cluster).map(d => ({ x: d[0][0], y: d[0][1] })),
        backgroundColor: colors[index],
        pointRadius: 5
    }));

    dataSetsGraph.push({
        label: 'Centroids',
        data: Array.from(clusters).map(value => ({ x: value[0], y: value[1] })),
        backgroundColor: '#ff0000',
        pointRadius: 10
    });

    console.log('Datasets', dataSetsGraph);

    chart = new Chart(chartDom, {
        type: 'scatter',
        data: {
            datasets: dataSetsGraph
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: 'KMeans Clustering (2D)'
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                },
                y: {
                    type: 'linear',
                    position: 'left'
                }
            }
        }
    });
    chrtCntainer.style.display = 'block';
    treeCntainer.style.display = 'none';
}

function handleNeuralNetwork() {
    alert('Neural Network algorithm is not implemented yet.');
}

function handleKNN() {
    alert('K-Nearest Neighbors algorithm is not implemented yet.');
}

function showNaiveBayesPrediction(prediction) {
    alert('Naive Bayes algorithm is not implemented yet.');
}

function showNeuralNetworkResults(nn, predictData) {
    alert('Neural Network algorithm is not implemented yet.');
}

function showKNNResults(euclideanDistances, manhattanDistances) {
    alert('K-Nearest Neighbors algorithm is not implemented yet.');
}

// Initialize the application
init();