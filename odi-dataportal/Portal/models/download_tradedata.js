var data = source.data;
var filetext = 'Country,Year,Food,OresMetals,Fuel,Manufactured,RawAgriculture\n';
for (i=0; i < data['Country'].length; i++) {
    var currRow = [data['Country'][i].toString(),
                   data['Year'][i].toString(),
		   data['Food Exports'][i].toString(),
		   data['Ores and Metals'][i].toString(),
		   data['Fuel Exports'][i].toString(),
		   data['Manufactured Goods'][i].toString(),
                   data['Raw Agriculture Products'][i].toString().concat('\n')];

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
