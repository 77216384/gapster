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

$('#buddhism').click(function(e){
  e.preventDefault();
  $('#questionLoading').show();
  $.ajax({
    type: "GET",
    url: "/fetch_article/buddhism",
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

$('#lincoln').click(function(e){
  e.preventDefault();
  $('#questionLoading').show();
  $.ajax({
    type: "GET",
    url: "/fetch_article/lincoln",
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

$('#snoop').click(function(e){
  e.preventDefault();
  $('#questionLoading').show();
  $.ajax({
    type: "GET",
    url: "/fetch_article/snoop",
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

$('#wormhole').click(function(e){
  e.preventDefault();
  $('#questionLoading').show();
  $.ajax({
    type: "GET",
    url: "/fetch_article/wormhole",
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