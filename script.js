const generateButton = document.getElementById('generateButton');
const resultSection = document.getElementById('result');

generateButton.addEventListener('click', () => {
    const featuresInput = document.getElementById('features').value;
    const dataTypesInput = document.getElementById('dataTypes').value;
    const sampleDataInput = document.getElementById('sampleData').value;

    const features = featuresInput.split(',').map(feature => feature.trim());
    const dataTypes = dataTypesInput.split(',').map(dataType => dataType.trim());

    const myHeaders = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT',
        'Access-Control-Allow-Headers': 'Content-Type',
      };
    const myHeadersJSON = JSON.stringify(myHeaders);
  

    const requestBody = {
        features: features,
        dataTypes: dataTypes,
        sampleData: parseInt(sampleDataInput)
    };
    

    fetch('127.0.0.1/generate-dataset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
           
        },
        body: JSON.stringify(requestBody)
    })
    .then(response => response.json())
    .then(dataset => {
        // Display the generated dataset
        resultSection.innerHTML = JSON.stringify(dataset, null, 2);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
