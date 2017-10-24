$('#demoSubmit').click(function(){
	$.ajax({
  		type: "POST",
  		url: '/question/',
  		data: JSON.stringify({ "text": $('#demoText').val()}),
  		success: function(data){
  			$('#demoResult .question').text(data.question);
  			$('#demoResult .question').text(data.question);
  		},
  		dataType: 'json'
	});
});