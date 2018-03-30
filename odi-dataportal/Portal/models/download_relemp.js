var data = source.data;
var filetext = 'Country,Year,ChangeEmploymentShare,RelativeLabourProductivity\n';
for (i=0; i < data['countryname'].length; i++) {
    var currRow = [data['countryname'][i].toString(),
                   data['Industry'][i].toString(),
		   data['empave'][i].toString(),
                   data['relLP'][i].toString().concat('\n')];

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
