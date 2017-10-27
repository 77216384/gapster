$('#questionLoading').hide();

$('#napoleon').click(function(e){
  e.preventDefault();
  $('#questionLoading').show();
	$.ajax({
		type: "GET",
		url: "/fetch_article/napoleon",
		success: function(data){
      $.ajax({
        type: "POST",
        url: "/question",
        data: JSON.stringify(data),
        contentType: "application/json; charset=utf-8",
        success: function(data){
          $('#questionLoading').hide();
          $('#questionText').text(data.question);
          $('#choiceOne').text(data.answer);
          $('#choiceTwo').text(data.distractors[0]);
          $('#choiceThree').text(data.distractors[1]);
          $('#choiceFour').text(data.distractors[2]);
        },
        dataType: 'json'
      });
		}
	});
});