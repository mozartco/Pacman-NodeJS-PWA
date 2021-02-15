/*const express = require('express')
const app = express()
const port = 3000

app.get('/', (req, res) => {
  res.render('index.html')
})

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})*/

/////////////////////////////////////////////////////////////////////////////////

/*const express = require('express');
const app = express();
const port = 3000

app.get('/', function(request, response){
    response.sendFile('C:/Users/HARI/pwa-3/index.html');
});

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`)
  })*/

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//Working

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/*var http = require('http');
var fs = require('fs');

const PORT=8080; 

fs.readFile('./index.html', function (err, html) {

    if (err) throw err;    

    http.createServer(function(request, response) {  
        response.writeHeader(200, {"Content-Type": "text/html"});  
        response.write(html);  
        response.end();  
    }).listen(PORT);
});*/

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/*Working*/

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

var express = require('express');
var app = express();

// app.use(express.static(__dirname + '/'));
app.use(express.static('app'))
app.use(express.static('/'))
app.listen('8080');
console.log('working on 8080');