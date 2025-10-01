const express  = require('express');

const app      = express();
 

const path = require('path');
 

const SerialPort = require('serialport').SerialPort;

const sp = new SerialPort( {

  path:'COM3',

  baudRate: 115200
});

 

const port = 3000;

 

app.get('/up',function(req,res)

{

	sp.write('w\n\r', function(err){

		if (err) {

            return console.log('Error on write: ', err.message);

        }

        console.log('send: up ');

		res.writeHead(200, {'Content-Type': 'text/plain'});

		res.end('\n');

	});

});

 

app.get('/down',function(req,res)

{

	sp.write('s\n\r', function(err){

		if (err) {

            return console.log('Error on write: ', err.message);

        }

        console.log('send: down');

		res.writeHead(200, {'Content-Type': 'text/plain'});

		res.end('\n');

	}); 

});



app.get('/left',function(req,res)

{

	sp.write('a\n\r', function(err){

		if (err) {

            return console.log('Error on write: ', err.message);

        }

        console.log('send: left');

		res.writeHead(200, {'Content-Type': 'text/plain'});

		res.end('\n');

	}); 

});

app.get('/right',function(req,res)

{

	sp.write('d\n\r', function(err){

		if (err) {

            return console.log('Error on write: ', err.message);

        }

        console.log('send: right ');

		res.writeHead(200, {'Content-Type': 'text/plain'});

		res.end('\n');

	});

});

app.get('/init',function(req,res)

{

	sp.write('i\n\r', function(err){

		if (err) {

            return console.log('Error on write: ', err.message);

        }

        console.log('send: init ');

		res.writeHead(200, {'Content-Type': 'text/plain'});

		res.end('\n');

	});

});


 
 

app.use(express.static(__dirname + '/public'));

 

app.listen(port, function() {

    console.log('listening on *:' + port);

});
