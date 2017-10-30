$('#questionLoading').hide();
var answer = '';

$('#arts').click(function(e){
  e.preventDefault();
  $('#articleTitle h3').text('');
  $('#questionText').text('');
  $('#choiceOne').text('').attr('class', 'btn btn-outline-light choiceButton').show();
  $('#choiceTwo').text('').attr('class', 'btn btn-outline-light choiceButton').show();
  $('#choiceThree').text('').attr('class', 'btn btn-outline-light choiceButton').show();
  $('#choiceFour').text('').attr('class', 'btn btn-outline-light choiceButton').show();
  $('#questionLoading').show();
	$.ajax({
		type: "GET",
		url: "/topic",
		success: function(data){
      articleTitle = data.title;
      $.ajax({
        type: "POST",
        url: "/question",
        data: JSON.stringify(data),
        contentType: "application/json; charset=utf-8",
        success: function(data){
          $('#questionLoading').hide();
          $('#articleTitle h3').text(articleTitle);
          $('#questionText').text(data.question);
          answer = data.answer;
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

$('.choiceButton').click(function(e){
  e.preventDefault();
  if ($(e.target).text() == answer) {
    $(e.target).addClass('btn-success').removeClass('btn-outline-light');
    var buttons = $('.choiceButton');
    for (i = 0; i < buttons.length; i++) { 
      if ($(buttons[i]).text() != answer) {
        $(buttons[i]).fadeOut('slow');
      }
    }
  } else {
    $(e.target).addClass('btn-danger').removeClass('btn-outline-light');
  }
});