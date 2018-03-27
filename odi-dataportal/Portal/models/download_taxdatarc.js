var data = source.data;
var filetext = 'Country,Year,Resource,Income,GoodsService,Trade,Other\n';
for (i=0; i < data['Country'].length; i++) {
    var currRow = [data['Country'][i].toString(),
                   data['Year'][i].toString(),
		   data['Resource'][i].toString(),
		   data['Income'][i].toString(),
		   data['GandS'][i].toString(),
		   data['Trade'][i].toString(),
                   data['Other'][i].toString().concat('\n')];

    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = 'data_result.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}

else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}
