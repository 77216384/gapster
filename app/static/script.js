$('#questionLoading').hide();
$('#questionBox').hide();
var answer = '';

$('.dropdown-item').click(function(e){
  e.preventDefault();
  $('#questionBox').fadeOut();
  $('#articleTitle').text('');
  $('#questionText').text('');
  $('#choiceOne').text('').attr('class', 'btn btn-outline-primary choiceButton').show();
  $('#choiceTwo').text('').attr('class', 'btn btn-outline-primary choiceButton').show();
  $('#choiceThree').text('').attr('class', 'btn btn-outline-primary choiceButton').show();
  $('#choiceFour').text('').attr('class', 'btn btn-outline-primary choiceButton').show();
  $('#questionLoading').show();
	$.ajax({
		type: "GET",
		url: "/topic/"+e.target.id,
		success: function(data){
      articleTitle = data.title;
      $.ajax({
        type: "POST",
        url: "/question",
        data: JSON.stringify(data),
        contentType: "application/json; charset=utf-8",
        success: function(data){
          answer = data.answer;
          $('#questionLoading').hide();
          $('#questionBox').fadeIn();
          $('#articleTitle').text('Article Title: '+articleTitle);
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

$('.choiceButton').click(function(e){
  e.preventDefault();
  if ($(e.target).text() == answer) {
    $(e.target).addClass('btn-success').removeClass('btn-outline-primary');
    var buttons = $('.choiceButton');
    for (i = 0; i < buttons.length; i++) { 
      if ($(buttons[i]).text() != answer) {
        $(buttons[i]).fadeOut('slow');
      }
    }
  } else {
    $(e.target).addClass('btn-danger').removeClass('btn-outline-primary');
  }
});