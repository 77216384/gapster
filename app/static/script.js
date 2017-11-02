$('#questionLoading').hide();
$('#questionBox').hide();
var answer = '';

$('.dropdown-item').click(function(e){
  e.preventDefault();
  $('#questionBox').hide();
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
          var random_index = Math.floor(Math.random() * 4);
          var choice_set = [];
          choice_set[random_index] = data.answer;
          for (i = 0; i < 4; i++) {
            if (choice_set[i] == undefined){
              choice_set[i] = data.distractors.pop();
            }
          }
          $('#questionLoading').hide();
          $('#questionBox').fadeIn();
          $('#articleTitle').text('Article Title: '+articleTitle);
          $('#questionText').text(data.question);
          $('#choiceOne').text(choice_set[0]);
          $('#choiceTwo').text(choice_set[1]);
          $('#choiceThree').text(choice_set[2]);
          $('#choiceFour').text(choice_set[3]);
        },
        dataType: 'json'
      });
		}
	});
});

$('.fa-times-circle').click(function(e){
  $('#questionBox').hide();
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