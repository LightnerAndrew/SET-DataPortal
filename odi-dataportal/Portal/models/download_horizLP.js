var data = source.data;
var filetext = 'Country,Year,LPGrowthAg,LPGrowthManufacturing,LPGrowthMining,LPGrowthConstruction,LPGrowthOther,LPGrowthRetail,LPGrowthTransportation,LPGrowthWithinAgriculture,LPGrowthWithinMining,LPGrowthWithinManufacturing,LPGrowthWithinConstruction,LPGrowthWithinRetail,LPGrowthWithinTransportation,LPGrowthWithinOther,LPGrowthBetweenAgriculture,LPGrowthBetweenMining,LPGrowthBetweenManufacturing,LPGrowthBetweenConstruction,LPGrowthBetweenRetail,LPGrowthBetweenTransportation,LPGrowthBetweenOt\n';
for (i=0; i < data['Country'].length; i++) {
    var currRow = [data['Country'][i].toString(),
                   data['Year'][i].toString(),
                   data['LPGrowth Agriculture'][i].toString(), 
		   data['LPGrowth Manufacturing'][i].toString(), 
		   data['LPGrowth Mining'][i].toString(), 
		   data['LPGrowth Construction'][i].toString(), 
		   data['LPGrowth Other'][i].toString(), 
		   data['LPGrowth Retail'][i].toString(), 
		   data['LPGrowth Transportation'][i].toString(), 
		   data['Within Agriculture'][i].toString(), 
		   data['Within Mining'][i].toString(), 
		   data['Within Manufacturing'][i].toString(), 
		   data['Within Construction'][i].toString(), 
		   data['Within Retail'][i].toString(), 
		   data['Within Transportation'][i].toString(), 
		   data['Within Other'][i].toString(), 
		   data['Between Agriculture'][i].toString(), 
		   data['Between Mining'][i].toString(), 
		   data['Between Manufacturing'][i].toString(), 
		   data['Between Construction'][i].toString(), 
		   data['Between Retail'][i].toString(), 
		   data['Between Transportation'][i].toString(),
		   data['Between Transportation'][i].toString().concat('\n')];

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
